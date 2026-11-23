// qtrack: stella_vslam monocular tracking over a frame list, for the Quest (run over adb) or a desktop.
// Writes, per map, the camera trajectory (TUM format, camera-to-world) and the keyframe trajectory under maps/map_NN/;
// maps.json; per-frame timing, tracking state and map (frames.csv); and the process's CPU time and peak memory.
// When tracking has been lost for --restart-after seconds of video time (default 2; 0 = never), the map is saved and a
// new one is started, instead of waiting for the camera to return to a place the old map covers. The top-level
// frame_trajectory.txt and keyframe_trajectory.txt are the map with the most tracked frames, for single-map tools.
//
//   qtrack --config cam.yaml --vocab orb_vocab.fbow --frames dir/rgb.txt --out outdir
//          [--root dir] [--first N] [--last M] [--no-loop] [--realtime-fps F] [--restart-after S]
#include <iomanip>
#include <set>
#include <stella_vslam/data/keyframe.h>
#include <stella_vslam/data/landmark.h>
#include <stella_vslam/publish/map_publisher.h>
#include <stella_vslam/system.h>
#include <stella_vslam/config.h>
#include <stella_vslam/publish/frame_publisher.h>
#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>

#include "map_segments.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <sys/resource.h>
#include <sys/stat.h>

namespace {
struct Item {
    double t;
    std::string path;
};

std::string arg(int argc, char** argv, const std::string& key, const std::string& def = "") {
    for (int i = 1; i + 1 < argc; ++i)
        if (key == argv[i]) return argv[i + 1];
    return def;
}
bool flag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i)
        if (key == argv[i]) return true;
    return false;
}
std::string dirname_of(const std::string& p) {
    auto s = p.find_last_of("/\\");
    return s == std::string::npos ? "." : p.substr(0, s);
}
long peak_rss_kb() {  // VmHWM: the process's peak resident memory
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmHWM:", 0) == 0) return std::stol(line.substr(6));
    return -1;
}
double cpu_seconds() {
    rusage u{};
    getrusage(RUSAGE_SELF, &u);
    return u.ru_utime.tv_sec + u.ru_stime.tv_sec + 1e-6 * (u.ru_utime.tv_usec + u.ru_stime.tv_usec);
}
void copy_file(const std::string& from, const std::string& to) {
    std::ifstream in(from, std::ios::binary);
    std::ofstream o(to, std::ios::binary);
    o << in.rdbuf();
}
double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(q * (v.size() - 1) + 0.5))];
}
}  // namespace

int main(int argc, char** argv) {
    const std::string cfg_path = arg(argc, argv, "--config"), vocab = arg(argc, argv, "--vocab"),
                      list = arg(argc, argv, "--frames"), out = arg(argc, argv, "--out");
    if (cfg_path.empty() || vocab.empty() || list.empty() || out.empty()) {
        std::cerr << "usage: qtrack --config cam.yaml --vocab orb_vocab.fbow --frames rgb.txt --out dir "
                     "[--root dir] [--first N] [--last M] [--no-loop] [--realtime-fps F] [--restart-after S]\n";
        return 2;
    }
    const std::string root = arg(argc, argv, "--root", dirname_of(list));
    const long first = std::stol(arg(argc, argv, "--first", "0"));
    const long last = std::stol(arg(argc, argv, "--last", "-1"));
    const double realtime_fps = std::stod(arg(argc, argv, "--realtime-fps", "0"));
    const double restart_after = std::stod(arg(argc, argv, "--restart-after", "2.0"));

    std::vector<Item> items;
    {
        std::ifstream f(list);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            Item it;
            ss >> it.t >> it.path;
            if (!it.path.empty()) items.push_back(it);
        }
    }
    if (items.empty()) {
        std::cerr << "no frames in " << list << "\n";
        return 2;
    }

    auto cfg = std::make_shared<stella_vslam::config>(YAML::LoadFile(cfg_path), cfg_path);
    stella_vslam::system slam(cfg, vocab);
    slam.startup();
    if (flag(argc, argv, "--no-loop")) slam.disable_loop_detector();

    qtrack::MapSegmenter seg(restart_after);
    int restarts = 0;
    ::mkdir((out + "/maps").c_str(), 0755);
    auto map_dir = [&](int id) {
        char d[32];
        std::snprintf(d, sizeof d, "/maps/map_%02d", id);
        return out + d;
    };
    auto save_map = [&](int id) {   // the live map's trajectories, before stella forgets them
        while (slam.loop_BA_is_running()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const std::string dir = map_dir(id);
        ::mkdir(dir.c_str(), 0755);
        slam.save_frame_trajectory(dir + "/frame_trajectory.txt", "TUM");
        slam.save_keyframe_trajectory(dir + "/keyframe_trajectory.txt", "TUM");
        // The map's 3-D points, each with the times of the keyframes that observed it: the reconstruction uses them to
        // calibrate a keyframe's depth where too little camera motion lets nothing be triangulated (a turn on the spot).
        std::vector<std::shared_ptr<stella_vslam::data::landmark>> lms;
        std::set<std::shared_ptr<stella_vslam::data::landmark>> local_lms;
        slam.get_map_publisher()->get_landmarks(lms, local_lms);
        std::ofstream lf(dir + "/landmarks.txt");
        lf << "# x y z (the map's own frame and units), then the times of the keyframes that observed the point\n";
        lf << std::fixed;
        for (const auto& lm : lms) {
            if (!lm || lm->will_be_erased()) continue;
            std::vector<double> ts;
            for (const auto& ob : lm->get_observations()) {
                const auto kf = ob.first.lock();
                if (kf) ts.push_back(kf->timestamp_);
            }
            if (ts.size() < 2) continue;
            std::sort(ts.begin(), ts.end());
            const auto p = lm->get_pos_in_world();
            lf << std::setprecision(5) << p(0) << ' ' << p(1) << ' ' << p(2) << std::setprecision(4);
            for (double t : ts) lf << ' ' << t;
            lf << '\n';
        }
    };

    std::ofstream log(out + "/frames.csv");
    log << "index,t,load_ms,track_ms,state,pose,map\n";
    std::vector<double> track_ms;
    int n_tracking = 0, n_lost = 0, n_init = 0, lost_events = 0;
    std::string prev_state = "Initializing";
    const auto wall0 = std::chrono::steady_clock::now();
    const double cpu0 = cpu_seconds();
    const long end = (last < 0 || last >= static_cast<long>(items.size())) ? static_cast<long>(items.size()) - 1 : last;

    for (long i = std::max(0L, first); i <= end; ++i) {
        const auto a = std::chrono::steady_clock::now();
        const cv::Mat img = cv::imread(root + "/" + items[i].path, cv::IMREAD_UNCHANGED);
        const auto b = std::chrono::steady_clock::now();
        if (img.empty()) {
            std::cerr << "cannot read " << items[i].path << "\n";
            continue;
        }
        const auto pose = slam.feed_monocular_frame(img, items[i].t);
        const auto c = std::chrono::steady_clock::now();
        const std::string state = slam.get_frame_publisher()->get_tracking_state();
        const double load = std::chrono::duration<double, std::milli>(b - a).count();
        const double trk = std::chrono::duration<double, std::milli>(c - b).count();
        track_ms.push_back(trk);
        if (state == "Tracking") ++n_tracking;
        else if (state == "Lost") ++n_lost;
        else ++n_init;
        if (state == "Lost" && prev_state != "Lost") ++lost_events;
        prev_state = state;
        const int map_now = seg.current_id();
        const bool restart = seg.step(items[i].t, state);
        log << i << "," << std::fixed << items[i].t << "," << load << "," << trk << "," << state << "," << (pose ? 1 : 0) << ","
            << (state == "Tracking" ? map_now : -1) << "\n";
        if (restart) {   // lost for --restart-after seconds: keep this map, start a new one at the next frame
            save_map(seg.current_id());
            slam.request_reset();
            seg.on_restart();
            ++restarts;
        }
        if (realtime_fps > 0) {  // optional: pace input like a live camera
            const double budget = 1000.0 / realtime_fps - (load + trk);
            if (budget > 0) std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(budget));
        }
    }
    while (slam.loop_BA_is_running()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
    const double cpu = cpu_seconds() - cpu0;

    const bool save_last = seg.last_needs_saving();
    if (save_last) save_map(seg.current_id());
    seg.finish(save_last);
    slam.shutdown();

    // maps.json, and the largest saved map under the old file names
    int best = -1;
    long best_n = 0, saved = 0, discarded = 0;
    {
        std::ofstream mj(out + "/maps.json");
        mj << "[\n";
        const auto& M = seg.maps();
        for (size_t k = 0; k < M.size(); ++k) {
            const auto& m = M[k];
            if (m.status == "saved") {
                ++saved;
                if (m.tracked_frames > best_n) best_n = m.tracked_frames, best = m.id;
            }
            if (m.status == "discarded") ++discarded;
            char d[32];
            std::snprintf(d, sizeof d, "maps/map_%02d", m.id);
            mj << "  {\"map\": " << m.id << ", \"status\": \"" << m.status << "\", \"first_t\": " << m.first_t
               << ", \"last_t\": " << m.last_t << ", \"tracked_frames\": " << m.tracked_frames << ", \"dir\": \""
               << (m.status == "saved" ? std::string(d) : std::string()) << "\"}" << (k + 1 < M.size() ? "," : "") << "\n";
        }
        mj << "]\n";
    }
    if (best >= 0) {
        copy_file(map_dir(best) + "/frame_trajectory.txt", out + "/frame_trajectory.txt");
        copy_file(map_dir(best) + "/keyframe_trajectory.txt", out + "/keyframe_trajectory.txt");
    }

    const int n = n_tracking + n_lost + n_init;
    std::ofstream js(out + "/stats.json");
    js << "{\n  \"frames\": " << n << ",\n  \"tracking\": " << n_tracking << ",\n  \"lost\": " << n_lost
       << ",\n  \"initializing\": " << n_init << ",\n  \"lost_events\": " << lost_events
       << ",\n  \"tracked_share\": " << (n ? static_cast<double>(n_tracking) / n : 0.0)
       << ",\n  \"track_ms_median\": " << percentile(track_ms, 0.5) << ",\n  \"track_ms_p95\": " << percentile(track_ms, 0.95)
       << ",\n  \"wall_s\": " << wall << ",\n  \"cpu_s\": " << cpu << ",\n  \"peak_rss_mb\": " << peak_rss_kb() / 1024.0
       << ",\n  \"loop_detector\": " << (flag(argc, argv, "--no-loop") ? "false" : "true")
       << ",\n  \"restart_after_s\": " << restart_after << ",\n  \"map_restarts\": " << restarts
       << ",\n  \"maps_saved\": " << saved << ",\n  \"maps_discarded\": " << discarded << ",\n  \"largest_map\": " << best << "\n}\n";
    std::cout << "frames " << n << ", tracking " << n_tracking << ", lost " << n_lost << " (" << lost_events
              << " events), " << saved << " map(s) saved, " << restarts << " restart(s), median " << percentile(track_ms, 0.5)
              << " ms, peak " << peak_rss_kb() / 1024.0 << " MB\n";
    return 0;
}
