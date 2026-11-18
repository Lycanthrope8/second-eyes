// qrecon_quest: the Quest-only reconstruction after tracking, run on the headset (or a desktop, for tests).
// Anchors the tracker's trajectory with ChArUco boards, picks keyframes, predicts depth (an ONNX model, or an oracle
// built from reference depth for tests), calibrates it in inverse depth against triangulated landmarks, keeps only
// depth that neighbouring keyframes agree with, and fuses it into one bounded, coloured point cloud.
// Same stages and parameters as the desktop harness (qrecon/), so the two can be compared on the same input.
//
//   qrecon_quest --config recon.json --out DIR
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#if (CV_VERSION_MAJOR > 4) || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
#include <opencv2/objdetect/charuco_detector.hpp>
#define QRECON_NEW_ARUCO 1
#else
#include <opencv2/aruco/charuco.hpp>
#define QRECON_NEW_ARUCO 0
#endif
#ifdef QRECON_WITH_ORT
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/resource.h>

using json = nlohmann::json;
using Mat4 = Eigen::Matrix4d;
using Mat3 = Eigen::Matrix3d;
using Vec3 = Eigen::Vector3d;

namespace {
// ---------------------------------------------------------------- small utilities
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
long peak_rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string l;
    while (std::getline(f, l))
        if (l.rfind("VmHWM:", 0) == 0) return std::stol(l.substr(6));
    return -1;
}
double cpu_s() {
    rusage u{};
    getrusage(RUSAGE_SELF, &u);
    return u.ru_utime.tv_sec + u.ru_stime.tv_sec + 1e-6 * (u.ru_utime.tv_usec + u.ru_stime.tv_usec);
}
double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(q * (v.size() - 1) + 0.5))];
}
std::string dir_of(const std::string& p) {
    auto s = p.find_last_of("/\\");
    return s == std::string::npos ? "." : p.substr(0, s);
}
struct Item {
    double t;
    std::string path;
};
std::vector<Item> read_list(const std::string& path) {
    std::vector<Item> v;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::string l;
    while (std::getline(f, l)) {
        if (l.empty() || l[0] == '#') continue;
        for (auto& c : l) if (c == ',') c = ' ';
        std::istringstream ss(l);
        Item it;
        ss >> it.t >> it.path;
        if (!it.path.empty()) v.push_back(it);
    }
    return v;
}
void read_tum(const std::string& path, std::vector<double>& times, std::vector<Mat4>& poses) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::string l;
    while (std::getline(f, l)) {
        if (l.empty() || l[0] == '#') continue;
        std::istringstream ss(l);
        double t, x, y, z, qx, qy, qz, qw;
        if (!(ss >> t >> x >> y >> z >> qx >> qy >> qz >> qw)) continue;
        Eigen::Quaterniond q(qw, qx, qy, qz);
        q.normalize();
        Mat4 T = Mat4::Identity();
        T.block<3, 3>(0, 0) = q.toRotationMatrix();
        T.block<3, 1>(0, 3) = Vec3(x, y, z);
        times.push_back(t);
        poses.push_back(T);
    }
}
int nearest(const std::vector<double>& times, double t, double max_dt) {
    if (times.empty()) return -1;
    auto it = std::lower_bound(times.begin(), times.end(), t);
    int j = static_cast<int>(it - times.begin());
    int best = -1;
    double bd = 1e18;
    for (int k : {j - 1, j}) {
        if (k < 0 || k >= static_cast<int>(times.size())) continue;
        double d = std::abs(times[k] - t);
        if (d < bd) bd = d, best = k;
    }
    return bd <= max_dt ? best : -1;
}
Mat4 inv(const Mat4& T) {
    Mat4 I = Mat4::Identity();
    I.block<3, 3>(0, 0) = T.block<3, 3>(0, 0).transpose();
    I.block<3, 1>(0, 3) = -T.block<3, 3>(0, 0).transpose() * T.block<3, 1>(0, 3);
    return I;
}
double rot_angle_deg(const Mat3& A, const Mat3& B) {
    double c = ((A.transpose() * B).trace() - 1) / 2;
    return std::acos(std::max(-1.0, std::min(1.0, c))) * 180.0 / M_PI;
}
struct Camera {
    Mat3 K;
    cv::Mat Kcv, dist;
    int w = 0, h = 0;
};
Camera camera_from(const json& j) {
    Camera c;
    c.Kcv = cv::Mat(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r)
        for (int k = 0; k < 3; ++k) c.K(r, k) = c.Kcv.at<double>(r, k) = j.at("K")[r][k].get<double>();
    std::vector<double> d = j.value("dist", std::vector<double>{0, 0, 0, 0, 0});
    c.dist = cv::Mat(d, true).reshape(1, 1);
    c.w = j.at("size")[0];
    c.h = j.at("size")[1];
    return c;
}

// ---------------------------------------------------------------- ChArUco boards
int dictionary_id(const std::string& n) {
    static const std::map<std::string, int> m = {
        {"DICT_4X4_50", cv::aruco::DICT_4X4_50}, {"DICT_4X4_100", cv::aruco::DICT_4X4_100}, {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
        {"DICT_5X5_50", cv::aruco::DICT_5X5_50}, {"DICT_5X5_100", cv::aruco::DICT_5X5_100}, {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
        {"DICT_6X6_50", cv::aruco::DICT_6X6_50}, {"DICT_6X6_100", cv::aruco::DICT_6X6_100}, {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
        {"DICT_APRILTAG_36h11", cv::aruco::DICT_APRILTAG_36h11}};
    auto it = m.find(n);
    if (it == m.end()) throw std::runtime_error("unsupported dictionary " + n);
    return it->second;
}
struct Board {
#if QRECON_NEW_ARUCO
    cv::Ptr<cv::aruco::CharucoBoard> board;
    cv::Ptr<cv::aruco::CharucoDetector> det;
#else
    cv::Ptr<cv::aruco::CharucoBoard> board;
    cv::Ptr<cv::aruco::Dictionary> dict;
    cv::Ptr<cv::aruco::DetectorParameters> params;
#endif
    std::vector<cv::Point3f> corners;
};
Board make_board(const json& b) {
    Board B;
    const int sx = b.at("squares_x"), sy = b.at("squares_y"), first = b.value("first_marker_id", 0);
    const float sq = b.at("square_length_m").get<float>(), mk = b.at("marker_length_m").get<float>();
    std::vector<int> ids(sx * sy / 2);
    std::iota(ids.begin(), ids.end(), first);
#if QRECON_NEW_ARUCO
    auto d = cv::aruco::getPredefinedDictionary(dictionary_id(b.at("dictionary")));
    B.board = cv::makePtr<cv::aruco::CharucoBoard>(cv::Size(sx, sy), sq, mk, d, ids);
    B.board->setLegacyPattern(b.value("legacy_pattern", false));
    B.det = cv::makePtr<cv::aruco::CharucoDetector>(*B.board);
    B.corners = B.board->getChessboardCorners();
#else
    if (b.value("legacy_pattern", false)) throw std::runtime_error("legacy pattern needs OpenCV 4.7 or later");
    B.dict = cv::aruco::getPredefinedDictionary(dictionary_id(b.at("dictionary")));
    B.board = cv::aruco::CharucoBoard::create(sx, sy, sq, mk, B.dict);
    B.board->ids = ids;
    B.params = cv::aruco::DetectorParameters::create();
    B.corners = B.board->chessboardCorners;
#endif
    return B;
}
bool detect(const Board& B, const cv::Mat& gray, int min_corners, std::vector<int>& ids, std::vector<cv::Point2f>& px) {
    std::vector<cv::Point2f> cc;
    std::vector<int> ci;
#if QRECON_NEW_ARUCO
    B.det->detectBoard(gray, cc, ci);
#else
    std::vector<std::vector<cv::Point2f>> mc;
    std::vector<int> mi;
    cv::aruco::detectMarkers(gray, B.dict, mc, mi, B.params);
    if (!mi.empty()) cv::aruco::interpolateCornersCharuco(mc, mi, gray, B.board, cc, ci);
#endif
    if (static_cast<int>(ci.size()) < min_corners) return false;
    ids = ci;
    px = cc;
    return true;
}
std::vector<Eigen::Vector2d> undistort(const std::vector<cv::Point2f>& px, const Camera& c) {
    std::vector<cv::Point2f> out;
    cv::undistortPoints(px, out, c.Kcv, c.dist, cv::noArray(), c.Kcv);
    std::vector<Eigen::Vector2d> v;
    for (auto& p : out) v.emplace_back(p.x, p.y);
    return v;
}
Eigen::Matrix<double, 3, 4> projection(const Mat3& K, const Mat4& T_wc) {
    Mat4 Tcw = inv(T_wc);
    return K * Tcw.block<3, 4>(0, 0);
}
Vec3 triangulate(const Mat3& K, const std::vector<Mat4>& poses, const std::vector<Eigen::Vector2d>& px) {
    Eigen::MatrixXd A(2 * poses.size(), 4);
    for (size_t i = 0; i < poses.size(); ++i) {
        auto P = projection(K, poses[i]);
        A.row(2 * i) = px[i].x() * P.row(2) - P.row(0);
        A.row(2 * i + 1) = px[i].y() * P.row(2) - P.row(1);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X = svd.matrixV().col(3);
    return X.head<3>() / X(3);
}
struct Obs {
    std::vector<int> ids;
    std::vector<Eigen::Vector2d> px;  // undistorted
    Mat4 pose;
    double t = 0.0;                   // board frame's time, for the per-frame report
};
std::map<int, Vec3> corners_in_frame(const std::vector<Obs>& obs, const Mat3& K, const json& cfg) {
    std::map<int, std::vector<std::pair<int, Eigen::Vector2d>>> seen;
    for (size_t f = 0; f < obs.size(); ++f)
        for (size_t k = 0; k < obs[f].ids.size(); ++k) seen[obs[f].ids[k]].push_back({static_cast<int>(f), obs[f].px[k]});
    std::map<int, Vec3> pts;
    for (auto& [cid, lst] : seen) {
        if (static_cast<int>(lst.size()) < cfg.at("min_views_per_corner").get<int>()) continue;
        std::vector<Mat4> P;
        std::vector<Eigen::Vector2d> uv;
        for (auto& [f, p] : lst) P.push_back(obs[f].pose), uv.push_back(p);
        Vec3 X = triangulate(K, P, uv);
        double min_cos = 1;
        std::vector<Vec3> rays;
        for (auto& T : P) rays.push_back((T.block<3, 1>(0, 3) - X).normalized());
        for (auto& a : rays)
            for (auto& b : rays) min_cos = std::min(min_cos, a.dot(b));
        if (std::acos(std::max(-1.0, std::min(1.0, min_cos))) * 180.0 / M_PI >= cfg.at("min_parallax_deg").get<double>()) pts[cid] = X;
    }
    return pts;
}
struct Sim3 {
    double s = 1;
    Mat3 R = Mat3::Identity();
    Vec3 t = Vec3::Zero();
    Mat4 apply(const Mat4& T) const {
        Mat4 o = T;
        o.block<3, 3>(0, 0) = R * T.block<3, 3>(0, 0);
        o.block<3, 1>(0, 3) = s * R * T.block<3, 1>(0, 3) + t;
        return o;
    }
};
Sim3 umeyama_fit(const std::vector<Vec3>& src, const std::vector<Vec3>& dst) {
    Eigen::Matrix3Xd a(3, src.size()), b(3, dst.size());
    for (size_t i = 0; i < src.size(); ++i) a.col(i) = src[i], b.col(i) = dst[i];
    Mat4 T = Eigen::umeyama(a, b, true);
    Sim3 S;
    S.s = T.block<3, 1>(0, 0).norm();
    S.R = T.block<3, 3>(0, 0) / S.s;
    S.t = T.block<3, 1>(0, 3);
    return S;
}
json fit_board(const std::vector<Obs>& obs, const Board& B, const Mat3& K, const json& cfg, Sim3* out, bool holdout) {
    json r;
    if (static_cast<int>(obs.size()) < cfg.at("min_views").get<int>()) {
        r["ok"] = false;
        r["reason"] = "board seen in too few frames";
        r["frames"] = obs.size();
        return r;
    }
    const int k = cfg.at("holdout_every");
    std::vector<Obs> fit, held;
    for (size_t i = 0; i < obs.size(); ++i)
        ((holdout && k > 0 && static_cast<int>(obs.size()) >= 2 * cfg.at("min_views").get<int>() && i % k == 1) ? held : fit).push_back(obs[i]);
    auto pts = corners_in_frame(fit, K, cfg);
    r["frames"] = obs.size();
    r["held_out_frames"] = held.size();
    r["triangulated_corners"] = pts.size();
    if (static_cast<int>(pts.size()) < cfg.at("min_corners").get<int>()) {
        r["ok"] = false;
        r["reason"] = "board views are not translated enough to triangulate its corners";
        return r;
    }
    std::vector<Vec3> src, dst;
    for (auto& [id, X] : pts) {
        src.push_back(X);
        auto c = B.corners.at(id);
        dst.emplace_back(c.x, c.y, c.z);
    }
    Sim3 S = umeyama_fit(src, dst);
    double ss = 0, mx = 0;
    for (size_t i = 0; i < src.size(); ++i) {
        double e = (S.s * S.R * src[i] + S.t - dst[i]).norm();
        ss += e * e;
        mx = std::max(mx, e);
    }
    r["scale"] = S.s;
    r["corner_fit_rms_m"] = std::sqrt(ss / src.size());
    r["corner_fit_max_m"] = mx;
    r["ok"] = r["corner_fit_rms_m"].get<double>() <= cfg.at("max_corner_fit_rms_m").get<double>();
    if (!r["ok"].get<bool>()) r["reason"] = "triangulated corners do not fit the measured board";
    // Reprojection of the measured board through the anchored path: in pixels, and as the distance at the board that
    // the pixel error amounts to (pixels x depth / focal length), which is comparable with corner_fit_rms_m.
    auto reproject = [&](const std::vector<Obs>& views, std::vector<double>& px_all, std::vector<double>& m_all, json* detail) {
        for (auto& o : views) {
            Mat4 Tb = S.apply(o.pose), Tcb = inv(Tb);
            std::vector<double> e_px, e_m, depth;
            for (size_t i = 0; i < o.ids.size(); ++i) {
                auto c = B.corners.at(o.ids[i]);
                Vec3 pc = Tcb.block<3, 3>(0, 0) * Vec3(c.x, c.y, c.z) + Tcb.block<3, 1>(0, 3);
                Eigen::Vector2d uv(K(0, 0) * pc.x() / pc.z() + K(0, 2), K(1, 1) * pc.y() / pc.z() + K(1, 2));
                const double e = (uv - o.px[i]).norm();
                e_px.push_back(e);
                e_m.push_back(e * pc.z() / K(0, 0));
                depth.push_back(pc.z());
            }
            px_all.insert(px_all.end(), e_px.begin(), e_px.end());
            m_all.insert(m_all.end(), e_m.begin(), e_m.end());
            if (detail)
                detail->push_back({{"t", o.t}, {"corners", o.ids.size()}, {"distance_m", pct(depth, 0.5)},
                                   {"median_px", pct(e_px, 0.5)}, {"median_m", pct(e_m, 0.5)}});
        }
    };
    {
        std::vector<double> px_fit, m_fit;
        reproject(fit, px_fit, m_fit, nullptr);
        if (!px_fit.empty()) r["fit_reprojection_px"] = {{"median", pct(px_fit, 0.5)}, {"p95", pct(px_fit, 0.95)}};
    }
    if (!held.empty()) {
        std::vector<double> errs, errs_m;
        json detail = json::array();
        reproject(held, errs, errs_m, &detail);
        r["held_out_reprojection_px"] = {{"median", pct(errs, 0.5)}, {"p95", pct(errs, 0.95)}};
        r["held_out_error_m"] = {{"median", pct(errs_m, 0.5)}, {"p95", pct(errs_m, 0.95)}};
        r["held_out_detail"] = detail;
        // The limit is either in metres at the board (max_held_out_m) or in pixels (max_held_out_px), whichever the
        // configuration names; metres take precedence.
        const bool metric = cfg.contains("max_held_out_m");
        const double value = metric ? pct(errs_m, 0.5) : pct(errs, 0.5);
        const double limit = metric ? cfg.at("max_held_out_m").get<double>() : cfg.at("max_held_out_px").get<double>();
        r["held_out_limit"] = {{"value", limit}, {"unit", metric ? "m" : "px"}};
        if (value > limit) {
            r["ok"] = false;
            r["reason"] = "held-out views of the board disagree with the anchored trajectory";
        }
    }
    if (out) *out = S;
    return r;
}

// ---------------------------------------------------------------- landmarks, depth, calibration, consistency, fusion
struct KF {
    int frame;          // index in the frames list
    double t;
    Mat4 T;             // metric camera-to-world
    cv::Mat bgr, gray;
    std::vector<cv::KeyPoint> kp;
    cv::Mat desc;
    cv::Mat rel, depth, filtered;  // network output, calibrated metric depth, after the cross-view check
    bool calibrated = false;
    json rec;
};
void landmarks_for(int i, std::vector<KF>& kfs, const Mat3& K, const json& cfg, std::vector<Eigen::Vector2d>& px, std::vector<double>& z) {
    cv::BFMatcher bf(cv::NORM_HAMMING);
    for (int off : cfg.at("neighbour_offsets").get<std::vector<int>>()) {
        int j = i + off;
        if (j < 0 || j >= static_cast<int>(kfs.size()) || kfs[i].desc.empty() || kfs[j].desc.empty() || kfs[j].kp.size() < 2) continue;
        std::vector<std::vector<cv::DMatch>> knn;
        bf.knnMatch(kfs[i].desc, kfs[j].desc, knn, 2);
        std::vector<cv::Point2f> a, b;
        for (auto& m : knn)
            if (m.size() == 2 && m[0].distance < cfg.at("ratio").get<double>() * m[1].distance)
                a.push_back(kfs[i].kp[m[0].queryIdx].pt), b.push_back(kfs[j].kp[m[0].trainIdx].pt);
        if (a.size() < 8) continue;
        cv::Mat Pa(3, 4, CV_64F), Pb(3, 4, CV_64F), X;
        const auto PA = projection(K, kfs[i].T), PB = projection(K, kfs[j].T);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c) Pa.at<double>(r, c) = PA(r, c), Pb.at<double>(r, c) = PB(r, c);
        cv::triangulatePoints(Pa, Pb, a, b, X);
        X.convertTo(X, CV_64F);
        Vec3 ca = kfs[i].T.block<3, 1>(0, 3), cb = kfs[j].T.block<3, 1>(0, 3);
        Mat4 Tia = inv(kfs[i].T), Tib = inv(kfs[j].T);
        for (int n = 0; n < X.cols; ++n) {
            Vec3 Xw(X.at<double>(0, n) / X.at<double>(3, n), X.at<double>(1, n) / X.at<double>(3, n), X.at<double>(2, n) / X.at<double>(3, n));
            Vec3 pa = Tia.block<3, 3>(0, 0) * Xw + Tia.block<3, 1>(0, 3), pb = Tib.block<3, 3>(0, 0) * Xw + Tib.block<3, 1>(0, 3);
            if (pa.z() <= cfg.at("min_depth_m").get<double>() || pb.z() <= cfg.at("min_depth_m").get<double>() ||
                pa.z() >= cfg.at("max_depth_m").get<double>())
                continue;
            Eigen::Vector2d ua(K(0, 0) * pa.x() / pa.z() + K(0, 2), K(1, 1) * pa.y() / pa.z() + K(1, 2));
            Eigen::Vector2d ub(K(0, 0) * pb.x() / pb.z() + K(0, 2), K(1, 1) * pb.y() / pb.z() + K(1, 2));
            if ((ua - Eigen::Vector2d(a[n].x, a[n].y)).norm() >= cfg.at("max_reproj_px").get<double>() ||
                (ub - Eigen::Vector2d(b[n].x, b[n].y)).norm() >= cfg.at("max_reproj_px").get<double>())
                continue;
            Vec3 ra = Xw - ca, rb = Xw - cb;
            double par = std::acos(std::max(-1.0, std::min(1.0, ra.dot(rb) / (ra.norm() * rb.norm() + 1e-12)))) * 180.0 / M_PI;
            if (par <= cfg.at("min_parallax_deg").get<double>()) continue;
            px.emplace_back(a[n].x, a[n].y);
            z.push_back(pa.z());
        }
    }
}
float sample(const cv::Mat& img, const Eigen::Vector2d& p) {
    int u = std::min(std::max(static_cast<int>(std::lround(p.x())), 0), img.cols - 1);
    int v = std::min(std::max(static_cast<int>(std::lround(p.y())), 0), img.rows - 1);
    return img.at<float>(v, u);
}
json fit_inverse_depth(const cv::Mat& rel, const std::vector<Eigen::Vector2d>& px, const std::vector<double>& z, const json& cfg,
                       unsigned seed, double& a_out, double& b_out) {
    json st;
    const int n = static_cast<int>(z.size());
    st["landmarks"] = n;
    if (n < cfg.at("min_landmarks").get<int>()) return json{{"ok", false}, {"reason", "too few landmarks"}, {"stats", st}};
    const int g = cfg.at("grid");
    std::set<std::pair<int, int>> cells;
    for (auto& p : px) cells.insert({std::min(g - 1, std::max(0, static_cast<int>(p.x() * g / rel.cols))), std::min(g - 1, std::max(0, static_cast<int>(p.y() * g / rel.rows)))});
    std::vector<double> zs = z;
    double spread = pct(zs, 0.9) / std::max(pct(zs, 0.1), 1e-6);
    st["cells"] = cells.size();
    st["spread"] = spread;
    if (static_cast<int>(cells.size()) < cfg.at("min_cells").get<int>() || spread < cfg.at("min_depth_spread").get<double>())
        return json{{"ok", false}, {"reason", "landmarks not spread over image and depth"}, {"stats", st}};
    std::mt19937 rng(seed);
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    int n_hold = std::max(1, static_cast<int>(std::lround(n * cfg.at("holdout_fraction").get<double>())));
    std::vector<int> hold(order.begin(), order.begin() + n_hold), fit(order.begin() + n_hold, order.end());
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) x[i] = sample(rel, px[i]), y[i] = 1.0 / z[i];
    std::vector<char> best;
    int best_n = -1;
    std::uniform_int_distribution<int> pick(0, static_cast<int>(fit.size()) - 1);
    const double inl_err = cfg.at("inlier_rel_err");
    for (int it = 0; it < cfg.at("ransac_iters").get<int>(); ++it) {
        int i = fit[pick(rng)], j = fit[pick(rng)];
        if (i == j || std::abs(x[i] - x[j]) < 1e-9) continue;
        double a = (y[i] - y[j]) / (x[i] - x[j]), b = y[i] - a * x[i];
        std::vector<char> inl(fit.size(), 0);
        int cnt = 0;
        for (size_t k = 0; k < fit.size(); ++k) {
            double ip = a * x[fit[k]] + b;
            if (ip > 1e-6 && std::abs(1.0 / ip - z[fit[k]]) / z[fit[k]] < inl_err) inl[k] = 1, ++cnt;
        }
        if (cnt > best_n) best_n = cnt, best = inl;
    }
    st["inliers"] = std::max(best_n, 0);
    if (best_n < cfg.at("min_inliers").get<int>()) return json{{"ok", false}, {"reason", "no consistent fit"}, {"stats", st}};
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int m = 0;
    for (size_t k = 0; k < fit.size(); ++k)
        if (best[k]) {
            double xx = x[fit[k]], yy = y[fit[k]];
            sx += xx, sy += yy, sxx += xx * xx, sxy += xx * yy, ++m;
        }
    double den = m * sxx - sx * sx;
    double a = std::abs(den) < 1e-18 ? 0 : (m * sxy - sx * sy) / den, b = (sy - a * sx) / m;
    std::vector<double> herr;
    bool bad = false;
    for (int i : hold) {
        double ip = a * x[i] + b;
        if (ip <= 1e-6) bad = true;
        else herr.push_back(std::abs(1.0 / ip - z[i]) / z[i]);
    }
    double hold_err = bad ? 1e9 : pct(herr, 0.5);
    double pos = 0;
    for (int r = 0; r < rel.rows; ++r)
        for (int c = 0; c < rel.cols; ++c) pos += (a * rel.at<float>(r, c) + b > 1e-6);
    pos /= static_cast<double>(rel.total());
    st["holdout_median_rel_err"] = hold_err;
    st["positive_share"] = pos;
    a_out = a, b_out = b;
    if (pos < cfg.at("min_positive_share").get<double>()) return json{{"ok", false}, {"reason", "fit gives non-positive depth over part of the image"}, {"stats", st}};
    if (hold_err > cfg.at("max_holdout_rel_err").get<double>()) return json{{"ok", false}, {"reason", "fit fails on held-out landmarks"}, {"stats", st}};
    return json{{"ok", true}, {"reason", ""}, {"a", a}, {"b", b}, {"stats", st}};
}
cv::Mat apply_inverse_depth(const cv::Mat& rel, double a, double b, double max_depth) {
    cv::Mat d(rel.size(), CV_32F, cv::Scalar(0));
    for (int r = 0; r < rel.rows; ++r)
        for (int c = 0; c < rel.cols; ++c) {
            double iv = a * rel.at<float>(r, c) + b;
            if (iv > 1.0 / max_depth) d.at<float>(r, c) = static_cast<float>(1.0 / iv);
        }
    return d;
}
cv::Mat filter_depth(int i, const std::map<int, KF*>& depths, const Mat3& K, const json& cfg, json& st) {
    const cv::Mat& d = depths.at(i)->depth;
    const Mat4& Ti = depths.at(i)->T;
    const double mrg = cfg.at("max_rel_depth_gradient"), mrd = cfg.at("max_rel_diff");
    cv::Mat out(d.size(), CV_32F, cv::Scalar(0));
    int n0 = 0, after_edges = 0, kept = 0, checked = 0, agree_checked = 0;
    std::vector<std::pair<int, KF*>> nb;
    for (int off : cfg.at("neighbour_offsets").get<std::vector<int>>())
        if (depths.count(i + off)) nb.push_back({i + off, depths.at(i + off)});
    for (int v = 0; v < d.rows; ++v)
        for (int u = 0; u < d.cols; ++u) {
            float z = d.at<float>(v, u);
            if (z <= 0) continue;
            ++n0;
            float gx = u > 0 ? std::abs(z - d.at<float>(v, u - 1)) : 0.f, gy = v > 0 ? std::abs(z - d.at<float>(v - 1, u)) : 0.f;
            if (std::max(gx, gy) / z >= mrg) continue;
            ++after_edges;
            Vec3 pc((u - K(0, 2)) / K(0, 0) * z, (v - K(1, 2)) / K(1, 1) * z, z);
            Vec3 pw = Ti.block<3, 3>(0, 0) * pc + Ti.block<3, 1>(0, 3);
            int votes = 0, seen = 0;
            for (auto& [j, kf] : nb) {
                Mat4 Tj = inv(kf->T);
                Vec3 q = Tj.block<3, 3>(0, 0) * pw + Tj.block<3, 1>(0, 3);
                if (q.z() <= 0) continue;
                long ui = std::lround(K(0, 0) * q.x() / q.z() + K(0, 2)), vi = std::lround(K(1, 1) * q.y() / q.z() + K(1, 2));
                if (ui < 0 || vi < 0 || ui >= kf->depth.cols || vi >= kf->depth.rows) continue;
                float dj = kf->depth.at<float>(vi, ui);
                if (dj <= 0) continue;
                ++seen;
                if (std::abs(dj - q.z()) <= mrd * q.z()) ++votes;
            }
            checked += seen > 0;
            agree_checked += (seen > 0 && votes > 0);
            bool ok = votes >= cfg.at("min_agreeing_views").get<int>() || (cfg.at("keep_unverified").get<bool>() && seen == 0);
            if (ok) out.at<float>(v, u) = z, ++kept;
        }
    st = {{"input_pixels", n0}, {"after_edges", after_edges}, {"kept", kept}, {"kept_share", n0 ? static_cast<double>(kept) / n0 : 0.0},
          {"checked_share", after_edges ? static_cast<double>(checked) / after_edges : 0.0},
          {"agree_share_of_checked", checked ? static_cast<double>(agree_checked) / checked : 0.0}};
    return out;
}
struct Voxel {
    double p[3] = {0, 0, 0}, c[3] = {0, 0, 0};
    long n = 0;
    int obs = 0;
};
struct VoxelMap {
    double v;
    size_t max_voxels;
    int min_obs;
    long refused = 0;
    std::unordered_map<uint64_t, Voxel> m;
    static uint64_t key(long x, long y, long z) {
        const long o = 1L << 20;
        return (static_cast<uint64_t>(x + o) << 42) | (static_cast<uint64_t>(y + o) << 21) | static_cast<uint64_t>(z + o);
    }
    void integrate(const std::vector<Vec3>& pts, const std::vector<Vec3>& cols) {
        std::unordered_map<uint64_t, Voxel> local;
        for (size_t i = 0; i < pts.size(); ++i) {
            auto& vx = local[key(std::floor(pts[i].x() / v), std::floor(pts[i].y() / v), std::floor(pts[i].z() / v))];
            for (int k = 0; k < 3; ++k) vx.p[k] += pts[i][k], vx.c[k] += cols[i][k];
            vx.n++;
        }
        for (auto& [k, lv] : local) {
            auto it = m.find(k);
            if (it == m.end()) {
                if (m.size() >= max_voxels) { ++refused; continue; }
                lv.obs = 1;
                m.emplace(k, lv);
            } else {
                for (int q = 0; q < 3; ++q) it->second.p[q] += lv.p[q], it->second.c[q] += lv.c[q];
                it->second.n += lv.n;
                it->second.obs += 1;
            }
        }
    }
};
void write_ply(const std::string& path, const VoxelMap& vm, const Mat3& R) {
    std::vector<const Voxel*> keep;
    for (auto& [k, vx] : vm.m) if (vx.obs >= vm.min_obs) keep.push_back(&vx);
    std::ofstream f(path, std::ios::binary);
    f << "ply\nformat binary_little_endian 1.0\nelement vertex " << keep.size()
      << "\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    for (auto* vx : keep) {
        Vec3 p = R * Vec3(vx->p[0] / vx->n, vx->p[1] / vx->n, vx->p[2] / vx->n);
        float xyz[3] = {static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())};
        unsigned char rgb[3];
        for (int k = 0; k < 3; ++k) rgb[k] = static_cast<unsigned char>(std::min(255.0, std::max(0.0, std::round(vx->c[k] / vx->n))));
        f.write(reinterpret_cast<char*>(xyz), sizeof(xyz));
        f.write(reinterpret_cast<char*>(rgb), sizeof(rgb));
    }
}

// ---------------------------------------------------------------- depth providers
struct DepthProvider {
    virtual ~DepthProvider() = default;
    virtual cv::Mat predict(const KF& kf) = 0;  // CV_32F relative inverse depth, frame size
    virtual json info() const = 0;
};
struct Oracle : DepthProvider {  // tests only: reference depth turned into an affine inverse depth
    std::vector<Item> list;
    std::vector<double> times;
    std::string root;
    double factor, a, b;
    explicit Oracle(const json& c) {
        list = read_list(c.at("depth_list"));
        root = dir_of(c.at("depth_list").get<std::string>());
        for (auto& it : list) times.push_back(it.t);
        factor = c.at("depth_factor"), a = c.at("a"), b = c.at("b");
    }
    cv::Mat predict(const KF& kf) override {
        int j = nearest(times, kf.t, 0.02);
        if (j < 0) throw std::runtime_error("oracle: no depth near a keyframe");
        cv::Mat d16 = cv::imread(root + "/" + list[j].path, cv::IMREAD_UNCHANGED), rel(d16.size(), CV_32F);
        for (int r = 0; r < d16.rows; ++r)
            for (int c = 0; c < d16.cols; ++c) {
                double z = d16.at<uint16_t>(r, c) / factor;
                rel.at<float>(r, c) = z > 0 ? static_cast<float>((1.0 / z - b) / a) : static_cast<float>((0.0 - b) / a);
            }
        return rel;
    }
    json info() const override { return {{"name", "oracle"}}; }
};
#ifdef QRECON_WITH_ORT
struct OnnxDepth : DepthProvider {  // Depth Anything V2 Small exported by scripts/export_dav2_onnx.py
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "qrecon"};
    Ort::Session session{nullptr};
    std::string model;
    int in_h = 0, in_w = 0;
    std::string in_name, out_name;
    double ms_total = 0;
    explicit OnnxDepth(const json& c) {
        model = c.at("model");
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(c.value("threads", 2));
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = Ort::Session(env, model.c_str(), so);
        Ort::AllocatorWithDefaultOptions al;
        in_name = session.GetInputNameAllocated(0, al).get();
        out_name = session.GetOutputNameAllocated(0, al).get();
        auto shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 4 || shape[2] <= 0 || shape[3] <= 0) throw std::runtime_error("the depth model needs a fixed input shape [1,3,H,W]");
        in_h = static_cast<int>(shape[2]), in_w = static_cast<int>(shape[3]);
    }
    cv::Mat predict(const KF& kf) override {
        cv::Mat rgb, rgbf, f;  // as Depth Anything V2: to float in [0,1] first, then cubic resize
        cv::cvtColor(kf.bgr, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgbf, CV_32F, 1.0 / 255.0);
        cv::resize(rgbf, f, cv::Size(in_w, in_h), 0, 0, cv::INTER_CUBIC);
        const float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
        std::vector<float> chw(3 * in_h * in_w);
        for (int y = 0; y < in_h; ++y)
            for (int x = 0; x < in_w; ++x) {
                const cv::Vec3f p = f.at<cv::Vec3f>(y, x);
                for (int k = 0; k < 3; ++k) chw[(k * in_h + y) * in_w + x] = (p[k] - mean[k]) / stdv[k];
            }
        std::array<int64_t, 4> shp{1, 3, in_h, in_w};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value in = Ort::Value::CreateTensor<float>(mem, chw.data(), chw.size(), shp.data(), shp.size());
        const char* inn[] = {in_name.c_str()};
        const char* outn[] = {out_name.c_str()};
        auto out = session.Run(Ort::RunOptions{nullptr}, inn, &in, 1, outn, 1);
        auto oshape = out[0].GetTensorTypeAndShapeInfo().GetShape();
        const int oh = static_cast<int>(oshape[oshape.size() - 2]), ow = static_cast<int>(oshape.back());
        cv::Mat o(oh, ow, CV_32F, const_cast<float*>(out[0].GetTensorData<float>()));
        // back to the frame size, bilinear with corner alignment (as Depth Anything V2's infer_image)
        cv::Mat r(kf.bgr.rows, kf.bgr.cols, CV_32F);
        const double sy = (oh - 1.0) / std::max(1, r.rows - 1), sx = (ow - 1.0) / std::max(1, r.cols - 1);
        for (int y = 0; y < r.rows; ++y) {
            double fy = y * sy;
            int y0 = static_cast<int>(fy), y1 = std::min(y0 + 1, oh - 1);
            double wy = fy - y0;
            for (int x = 0; x < r.cols; ++x) {
                double fx = x * sx;
                int x0 = static_cast<int>(fx), x1 = std::min(x0 + 1, ow - 1);
                double wx = fx - x0;
                r.at<float>(y, x) = static_cast<float>((1 - wy) * ((1 - wx) * o.at<float>(y0, x0) + wx * o.at<float>(y0, x1)) +
                                                       wy * ((1 - wx) * o.at<float>(y1, x0) + wx * o.at<float>(y1, x1)));
            }
        }
        return r;
    }
    json info() const override { return {{"name", "onnx"}, {"model", model}, {"input", {in_h, in_w}}}; }
};
#endif
}  // namespace

int main(int argc, char** argv) {
    std::string cfg_path, out;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--config") cfg_path = argv[i + 1];
        if (std::string(argv[i]) == "--out") out = argv[i + 1];
    }
    if (cfg_path.empty() || out.empty()) {
        std::cerr << "usage: qrecon_quest --config recon.json --out DIR\n";
        return 2;
    }
    const double wall0 = now_s(), cpu0 = cpu_s();
    json C;
    std::ifstream(cfg_path) >> C;
    json report, stats;
    std::map<std::string, std::vector<double>> stage_ms;
    try {
        // ---- inputs
        auto frames = read_list(C["frames"]);
        const std::string froot = dir_of(C["frames"].get<std::string>());
        std::vector<double> tt;
        std::vector<Mat4> tp;
        read_tum(C["trajectory"], tt, tp);
        Camera cam = camera_from(C["camera"]);
        const double max_dt = C.value("max_pose_dt_s", 0.05);

        // ---- anchor
        Sim3 S;
        Mat3 Rup = Mat3::Identity();
        double t0 = now_s();
        if (C["anchor"].value("enabled", false)) {
            const json& A = C["anchor"];
            Camera bc = camera_from(A.at("camera"));
            auto blist = read_list(A.at("board_frames"));
            const std::string broot = dir_of(A.at("board_frames").get<std::string>());
            auto collect = [&](const Board& B, const std::vector<std::vector<double>>& windows, const Sim3* pre) {
                std::vector<Obs> obs;
                for (auto& it : blist) {
                    bool in = false;
                    for (auto& w : windows) in |= (it.t >= w[0] && it.t <= w[1]);
                    if (!in) continue;
                    int j = nearest(tt, it.t, max_dt);
                    if (j < 0) continue;
                    cv::Mat g = cv::imread(broot + "/" + it.path, cv::IMREAD_GRAYSCALE);
                    std::vector<int> ids;
                    std::vector<cv::Point2f> px;
                    if (g.empty() || !detect(B, g, A.at("params")["min_corners"], ids, px)) continue;
                    obs.push_back({ids, undistort(px, bc), pre ? pre->apply(tp[j]) : tp[j], it.t});
                }
                return obs;
            };
            // Board windows per board: {"ref01": [[a, b], ...], ...} with "board_order" (found by video_tools.py
            // board-spans); older configs name them ref01_start, ref01_end and ref02 under "windows".
            using Span = std::pair<std::string, std::vector<double>>;      // label, [start, end] in seconds
            std::vector<std::string> order;
            std::map<std::string, std::vector<Span>> bw;
            auto fmt1 = [](double x) { std::ostringstream o; o.setf(std::ios::fixed); o.precision(1); o << x; return o.str(); };
            if (A.contains("board_windows")) {
                for (auto& [k, v] : A.at("board_windows").items())
                    for (auto& x : v) {
                        auto sp = x.get<std::vector<double>>();
                        bw[k].push_back({k + " " + fmt1(sp[0]) + "-" + fmt1(sp[1]) + " s", sp});
                    }
                if (A.contains("board_order"))
                    for (auto& k : A.at("board_order")) order.push_back(k.get<std::string>());
            } else {
                const json& w = A.at("windows");
                auto legacy = [&](const std::string& board, const std::string& key) {
                    if (!w.contains(key) || w[key].empty()) return;
                    if (w[key][0].is_array())
                        for (auto& x : w[key]) bw[board].push_back({key, x.get<std::vector<double>>()});
                    else
                        bw[board].push_back({key, w[key].get<std::vector<double>>()});
                };
                legacy("ref01", "ref01_start");
                legacy("ref01", "ref01_end");
                legacy("ref02", "ref02");
            }
            for (auto& [k, v] : A.at("boards").items())
                if (std::find(order.begin(), order.end(), k) == order.end()) order.push_back(k);
            auto usable_board = [&](const std::string& b) { return bw.count(b) && !bw[b].empty() && A.at("boards").contains(b); };
            auto spans_of = [&](const std::string& b, int skip) {
                std::vector<std::vector<double>> v;
                if (bw.count(b))
                    for (int i = 0; i < static_cast<int>(bw[b].size()); ++i)
                        if (i != skip) v.push_back(bw[b][i].second);
                return v;
            };
            const bool best_effort = A.value("mode", std::string("strict")) == "best_effort";
            json warnings = json::array();
            std::string frame = "tracker", anchored_window;
            int anchored_index = -1;
            if (!best_effort) {
                // Strict: the first window of the preferred board must pass every check
                std::string b0;
                for (auto& k : order)
                    if (usable_board(k)) { b0 = k; break; }
                if (b0.empty()) throw std::runtime_error("anchor refused: no board window");
                Board B = make_board(A.at("boards")[b0]);
                report[b0] = fit_board(collect(B, {bw[b0][0].second}, nullptr), B, bc.K, A.at("params"), &S, true);
                if (!report[b0]["ok"].get<bool>()) throw std::runtime_error("anchor refused: " + report[b0]["reason"].get<std::string>());
                frame = b0, anchored_window = bw[b0][0].first, anchored_index = 0;
            } else {
                // Best effort: every window of every board, and each board's windows taken together, is fitted. The anchor
                // is the passing fit of the most preferred board (the tightest corner fit within it); if none passes, the
                // tightest corner fit of all. A failed quality check is recorded as a warning, not a refusal.
                struct Cand {
                    std::string board, label;
                    int index, rank;
                    Sim3 Q;
                    json r;
                };
                std::vector<Cand> cands;
                json attempts = json::array();
                for (int bi = 0; bi < static_cast<int>(order.size()); ++bi) {
                    const std::string b = order[bi];
                    if (!usable_board(b)) continue;
                    Board B = make_board(A.at("boards")[b]);
                    const int n = static_cast<int>(bw[b].size());
                    for (int i = 0; i <= n; ++i) {
                        if (i == n && n < 2) break;                      // "all windows" only when there are several
                        const auto spans = i < n ? std::vector<std::vector<double>>{bw[b][i].second} : spans_of(b, -1);
                        const std::string label = i < n ? bw[b][i].first : b + " all windows";
                        Sim3 Q;
                        json r = fit_board(collect(B, spans, nullptr), B, bc.K, A.at("params"), &Q, true);
                        json att = {{"board", b}, {"window", label}, {"ok", r["ok"]}, {"reason", r.value("reason", std::string())}};
                        if (r.contains("corner_fit_rms_m")) att["corner_fit_rms_m"] = r["corner_fit_rms_m"];
                        attempts.push_back(att);
                        if (r.contains("scale")) cands.push_back({b, label, i < n ? i : -1, bi, Q, r});
                    }
                }
                auto fit_of = [](const Cand& c) { return c.r.value("corner_fit_rms_m", 1e9); };
                const Cand* best = nullptr;
                for (const auto& c : cands)
                    if (c.r["ok"].get<bool>() && (!best || c.rank < best->rank || (c.rank == best->rank && fit_of(c) < fit_of(*best)))) best = &c;
                if (!best)
                    for (const auto& c : cands)
                        if (!best || fit_of(c) < fit_of(*best)) best = &c;
                if (best) {
                    S = best->Q, frame = best->board, anchored_window = best->label, anchored_index = best->index;
                    report[frame] = best->r;
                    if (!best->r["ok"].get<bool>())
                        warnings.push_back("anchored by " + frame + " (" + anchored_window + ") although " + best->r["reason"].get<std::string>());
                }
                report["anchor_attempts"] = attempts;
                if (frame == "tracker") warnings.push_back("no board could be fitted: the cloud is in the tracker's own units, unscaled");
            }
            report["frame"] = frame;
            report["anchored_by"] = {{"board", frame}, {"window", anchored_window}};
            auto second = [&](const Board& B, const std::vector<std::vector<double>>& wins, const std::string& what) {
                auto o = collect(B, wins, &S);
                Sim3 Q;
                json r = fit_board(o, B, bc.K, A.at("params"), &Q, false);
                if (best_effort && !r.contains("scale")) {
                    // The views could not be triangulated (typically a camera standing still at the board): take the board's
                    // pose from each view on its own (PnP) through that view's anchored camera pose, and average them.
                    std::vector<Mat4> Ts;
                    for (auto& ob : o) {
                        if (ob.ids.size() < 6) continue;
                        std::vector<cv::Point3f> P3;
                        std::vector<cv::Point2f> p2;
                        for (size_t i = 0; i < ob.ids.size(); ++i) {
                            auto c = B.corners.at(ob.ids[i]);
                            P3.emplace_back(float(c.x), float(c.y), float(c.z));
                            p2.emplace_back(float(ob.px[i].x()), float(ob.px[i].y()));
                        }
                        cv::Mat Kc = (cv::Mat_<double>(3, 3) << bc.K(0, 0), 0, bc.K(0, 2), 0, bc.K(1, 1), bc.K(1, 2), 0, 0, 1);
                        cv::Mat rv, tv, Rm;
                        if (!cv::solvePnP(P3, p2, Kc, cv::noArray(), rv, tv, false, cv::SOLVEPNP_IPPE)) continue;
                        cv::Rodrigues(rv, Rm);
                        Mat4 Tcb = Mat4::Identity();
                        for (int a = 0; a < 3; ++a) {
                            for (int b = 0; b < 3; ++b) Tcb(a, b) = Rm.at<double>(a, b);
                            Tcb(a, 3) = tv.at<double>(a);
                        }
                        Ts.push_back(ob.pose * Tcb);                         // frame <- camera <- board
                    }
                    if (Ts.size() >= 3) {
                        Mat3 Msum = Mat3::Zero();
                        for (auto& T : Ts) Msum += T.block<3, 3>(0, 0);
                        Eigen::JacobiSVD<Mat3> svd(Msum, Eigen::ComputeFullU | Eigen::ComputeFullV);
                        Mat3 Rm = svd.matrixU() * svd.matrixV().transpose();
                        if (Rm.determinant() < 0) {
                            Mat3 D = Mat3::Identity(); D(2, 2) = -1;
                            Rm = svd.matrixU() * D * svd.matrixV().transpose();
                        }
                        Vec3 tm;
                        std::vector<double> spread;
                        for (int a = 0; a < 3; ++a) {
                            std::vector<double> v;
                            for (auto& T : Ts) v.push_back(T(a, 3));
                            tm(a) = pct(v, 0.5);
                        }
                        for (auto& T : Ts) spread.push_back((T.block<3, 1>(0, 3) - tm).norm());
                        Q.s = 1.0;
                        Q.R = Rm.transpose();                                // stored the way fit_board stores it
                        Q.t = -Q.R * tm;
                        r["scale"] = 1.0;
                        r["method"] = "single views (PnP), averaged";
                        r["views_used"] = Ts.size();
                        r["position_spread_m"] = pct(spread, 0.5);
                        r["reason"] = r.value("reason", std::string("too little camera motion to triangulate")) +
                                      "; pose taken from single views instead, boards assumed printed at the same scale";
                    }
                }
                const bool usable = r["ok"].get<bool>() || (best_effort && r.contains("scale"));
                if (usable) {
                    if (!r["ok"].get<bool>()) warnings.push_back(what + " used although " + r["reason"].get<std::string>());
                    r["scale_drift"] = 1.0 / Q.s - 1.0;
                    const Mat3 Rt = Q.R.transpose();
                    const Vec3 tt = -Rt * Q.t / Q.s;
                    Mat4 L = Mat4::Identity();
                    L.block<3, 3>(0, 0) = Rt;
                    L.block<3, 1>(0, 3) = tt;
                    const auto M = std::vector<std::vector<double>>{{L(0, 0), L(0, 1), L(0, 2), L(0, 3)}, {L(1, 0), L(1, 1), L(1, 2), L(1, 3)},
                                                                    {L(2, 0), L(2, 1), L(2, 2), L(2, 3)}, {0, 0, 0, 1}};
                    r["board_in_frame"] = M;
                    if (frame == "ref01") r["board_in_ref01"] = M;
                    r["sim3_board_in_frame"] = {{"s", 1.0 / Q.s},
                                                {"R", std::vector<std::vector<double>>{{Rt(0, 0), Rt(0, 1), Rt(0, 2)}, {Rt(1, 0), Rt(1, 1), Rt(1, 2)},
                                                                                       {Rt(2, 0), Rt(2, 1), Rt(2, 2)}}},
                                                {"t", std::vector<double>{tt(0), tt(1), tt(2)}}};
                    r["position_m"] = tt.norm();
                    r["rotation_deg"] = rot_angle_deg(Mat3::Identity(), Rt);
                }
                return r;
            };
            if (frame != "tracker") {
                // The anchoring board's other windows check closure; every other board seen gives a link
                const auto others = anchored_index >= 0 ? spans_of(frame, anchored_index) : std::vector<std::vector<double>>{};
                if (!others.empty()) report["closure_" + frame] = second(make_board(A.at("boards")[frame]), others, "the closure check");
                json links = json::object();
                for (auto& b : order) {
                    if (b == frame || !usable_board(b)) continue;
                    json r = second(make_board(A.at("boards")[b]), spans_of(b, -1), b + "'s pose");
                    r["note"] = "the relative scale assumes both boards are printed at their configured sizes";
                    links[b] = r;
                }
                report["links"] = links;
                if (frame == "ref01" && links.contains("ref02")) report["ref02"] = links["ref02"];        // older readers
                if (frame == "ref02" && links.contains("ref01")) report["ref01_link"] = links["ref01"];
                if (frame != "ref01") report["frame_note"] = frame + "'s frame, at its configured print scale";
            }
            if (!warnings.empty()) {
                report["warnings"] = warnings;
                for (auto& m : warnings) std::cerr << "qrecon_quest: warning: " << m.get<std::string>() << "\n";
            }
            stats["anchor_frame"] = frame;
            stats["anchor_warnings"] = warnings.size();
            if (frame != "tracker" && A.at("boards")[frame].value("mounting", "unknown") == "upright_level") Rup << 1, 0, 0, 0, 0, 1, 0, -1, 0;
        }
        stage_ms["anchor"].push_back(1000 * (now_s() - t0));
        {   // the anchored camera path, in the same TUM format as the tracker's
            std::ofstream f(out + "/anchored_trajectory.txt");
            f.precision(9);
            for (size_t j = 0; j < tp.size(); ++j) {
                Mat4 T = S.apply(tp[j]);
                Eigen::Quaterniond q(Mat3(T.block<3, 3>(0, 0)));
                f << std::fixed << tt[j] << " " << T(0, 3) << " " << T(1, 3) << " " << T(2, 3) << " " << q.x() << " " << q.y() << " "
                  << q.z() << " " << q.w() << "\n";
            }
        }
        if (C.value("anchor_only", false)) {
            report["ok"] = true;
            throw std::runtime_error("__anchor_only__");
        }

        // ---- keyframes on metric poses
        const json& KC = C["keyframes"];
        std::vector<KF> kfs;
        int last = -1;
        for (size_t f = 0; f < frames.size(); ++f) {
            int j = nearest(tt, frames[f].t, max_dt);
            if (j < 0) continue;
            Mat4 T = S.apply(tp[j]);
            if (last >= 0) {
                const KF& p = kfs.back();
                if (frames[f].t - p.t < KC.at("min_dt_s").get<double>()) continue;
                bool moved = (T.block<3, 1>(0, 3) - p.T.block<3, 1>(0, 3)).norm() >= KC.at("min_trans_m").get<double>();
                bool turned = rot_angle_deg(p.T.block<3, 3>(0, 0), T.block<3, 3>(0, 0)) >= KC.at("min_rot_deg").get<double>();
                if (!moved && !turned) continue;
            }
            KF k;
            k.frame = static_cast<int>(f), k.t = frames[f].t, k.T = T;
            kfs.push_back(k);
            last = static_cast<int>(f);
            if (C.contains("max_keyframes") && !C["max_keyframes"].is_null() && static_cast<int>(kfs.size()) >= C["max_keyframes"].get<int>()) break;
        }
        stats["keyframes"] = kfs.size();

        // ---- depth provider
        std::unique_ptr<DepthProvider> dp;
        if (kfs.empty()) throw std::runtime_error("no keyframes (tracking poses missing for every frame, or max_keyframes is 0)");
        if (C["depth"]["source"] == "oracle") dp = std::make_unique<Oracle>(C["depth"]["oracle"]);
#ifdef QRECON_WITH_ORT
        else if (C["depth"]["source"] == "onnx") dp = std::make_unique<OnnxDepth>(C["depth"]);
#endif
        else throw std::runtime_error("depth source not available in this build");
        report["depth_model"] = dp->info();

        // ---- streaming: landmarks lag 2, filtering lag 4, then fusion; bounded memory
        const json& FU = C["fusion"];
        VoxelMap vm{FU.at("voxel_m").get<double>(), FU.at("max_voxels").get<size_t>(), FU.at("min_observations").get<int>()};
        auto orb = cv::ORB::create(C["landmarks"]["orb_features"].get<int>());
        std::map<int, KF*> calibrated;
        const int n = static_cast<int>(kfs.size());
        auto load = [&](int i) {
            double a = now_s();
            kfs[i].bgr = cv::imread(froot + "/" + frames[kfs[i].frame].path, cv::IMREAD_COLOR);
            cv::cvtColor(kfs[i].bgr, kfs[i].gray, cv::COLOR_BGR2GRAY);
            orb->detectAndCompute(kfs[i].gray, cv::noArray(), kfs[i].kp, kfs[i].desc);
            stage_ms["load_orb"].push_back(1000 * (now_s() - a));
        };
        auto calibrate = [&](int i) {
            KF& k = kfs[i];
            std::vector<Eigen::Vector2d> px;
            std::vector<double> z;
            double a = now_s();
            landmarks_for(i, kfs, cam.K, C["landmarks"], px, z);
            stage_ms["landmarks"].push_back(1000 * (now_s() - a));
            a = now_s();
            k.rel = dp->predict(k);
            stage_ms["depth"].push_back(1000 * (now_s() - a));
            double ca = 0, cb = 0;
            json r = fit_inverse_depth(k.rel, px, z, C["calibration"], static_cast<unsigned>(k.frame), ca, cb);
            k.rec = {{"frame", k.frame}, {"t", k.t}, {"landmarks", z.size()}, {"calibrated", r["ok"]}, {"reason", r["reason"]}, {"calibration", r["stats"]}};
            if (r["ok"].get<bool>()) {
                k.depth = apply_inverse_depth(k.rel, ca, cb, FU.at("max_depth_m").get<double>());
                k.calibrated = true;
                calibrated[i] = &k;
            }
            k.rel.release();
        };
        auto fuse = [&](int i) {
            KF& k = kfs[i];
            if (!k.calibrated) return;
            json st;
            double a = now_s();
            k.filtered = filter_depth(i, calibrated, cam.K, C["consistency"], st);
            stage_ms["consistency"].push_back(1000 * (now_s() - a));
            k.rec["consistency"] = st;
            if (C.value("debug_dump", false)) {
                cv::imwrite(out + "/depth_" + std::to_string(k.frame) + ".tiff", k.depth);
                cv::imwrite(out + "/filtered_" + std::to_string(k.frame) + ".tiff", k.filtered);
            }
            a = now_s();
            std::vector<Vec3> pts, cols;
            const int stride = FU.at("pixel_stride");
            const double maxd = FU.at("max_depth_m");
            for (int v = 0; v < k.filtered.rows; v += stride)
                for (int u = 0; u < k.filtered.cols; u += stride) {
                    float z = k.filtered.at<float>(v, u);
                    if (z <= 0 || z > maxd) continue;
                    Vec3 pc((u - cam.K(0, 2)) / cam.K(0, 0) * z, (v - cam.K(1, 2)) / cam.K(1, 1) * z, z);
                    pts.push_back(k.T.block<3, 3>(0, 0) * pc + k.T.block<3, 1>(0, 3));
                    auto c = k.bgr.at<cv::Vec3b>(v, u);
                    cols.emplace_back(c[2], c[1], c[0]);
                }
            vm.integrate(pts, cols);
            stage_ms["fusion"].push_back(1000 * (now_s() - a));
        };
        auto release = [&](int i) {
            if (i < 0) return;
            KF& k = kfs[i];
            k.bgr.release(), k.gray.release(), k.desc.release(), k.kp.clear(), k.depth.release(), k.filtered.release();
            calibrated.erase(i);
        };
        for (int s = 0; s < n + 4; ++s) {
            if (s < n) load(s);
            if (s - 2 >= 0 && s - 2 < n) calibrate(s - 2);
            if (s - 4 >= 0 && s - 4 < n) fuse(s - 4);
            if (s - 6 >= 0) release(s - 6);
            if (s - 6 >= 0 && s - 6 < n) {}  // window of 7 keyframes is the most kept in memory
        }
        std::ofstream(out + "/keyframes.json") << json([&] { json a = json::array(); for (auto& k : kfs) a.push_back(k.rec); return a; }()).dump(1);
        int valid = 0;
        for (auto& k : kfs) valid += k.rec.value("calibrated", false) ? 1 : 0;
        write_ply(out + "/cloud.ply", vm, Mat3::Identity());
        if (!Rup.isIdentity()) write_ply(out + "/cloud_zup.ply", vm, Rup);
        long fused = 0;
        for (auto& [kk, vx] : vm.m) fused += vx.obs >= vm.min_obs;
        stats["valid_keyframes"] = valid;
        stats["valid_keyframe_share"] = n ? static_cast<double>(valid) / n : 0.0;
        stats["voxels"] = vm.m.size();
        stats["points"] = fused;
        stats["refused_new_voxels"] = vm.refused;
        report["ok"] = true;
    } catch (const std::exception& e) {
        if (std::string(e.what()) != "__anchor_only__") {
            report["ok"] = false;
            report["error"] = e.what();
            std::cerr << "qrecon_quest: " << e.what() << "\n";
        }
    }
    for (auto& [k, v] : stage_ms) stats["stage_ms"][k] = {{"median", pct(v, 0.5)}, {"p95", pct(v, 0.95)}, {"total", std::accumulate(v.begin(), v.end(), 0.0)}};
    stats["wall_s"] = now_s() - wall0;
    stats["cpu_s"] = cpu_s() - cpu0;
    stats["peak_rss_mb"] = peak_rss_kb() / 1024.0;
    stats["opencv"] = CV_VERSION;
    std::ofstream(out + "/anchor.json") << report.dump(1);
    std::ofstream(out + "/stats.json") << stats.dump(1);
    std::cout << stats.dump() << "\n";
    return report["ok"].get<bool>() ? 0 : 1;
}
