// qtrack: stella_vslam monocular tracking over a frame list, for the Quest (run over adb) or a desktop.
// Writes the camera trajectory (TUM format, camera-to-world), the keyframe trajectory, per-frame timing and
// tracking state, and the process's CPU time and peak memory.
//
//   qtrack --config cam.yaml --vocab orb_vocab.fbow --frames dir/rgb.txt --out outdir
//          [--root dir] [--first N] [--last M] [--no-loop] [--realtime-fps F]
#include <stella_vslam/system.h>
#include <stella_vslam/config.h>
#include <stella_vslam/publish/frame_publisher.h>
#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>

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
                     "[--root dir] [--first N] [--last M] [--no-loop] [--realtime-fps F]\n";
        return 2;
    }
    const std::string root = arg(argc, argv, "--root", dirname_of(list));
    const long first = std::stol(arg(argc, argv, "--first", "0"));
    const long last = std::stol(arg(argc, argv, "--last", "-1"));
    const double realtime_fps = std::stod(arg(argc, argv, "--realtime-fps", "0"));

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

    std::ofstream log(out + "/frames.csv");
    log << "index,t,load_ms,track_ms,state,pose\n";
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
        log << i << "," << std::fixed << items[i].t << "," << load << "," << trk << "," << state << "," << (pose ? 1 : 0) << "\n";
        if (realtime_fps > 0) {  // optional: pace input like a live camera
            const double budget = 1000.0 / realtime_fps - (load + trk);
            if (budget > 0) std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(budget));
        }
    }
    while (slam.loop_BA_is_running()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
    const double cpu = cpu_seconds() - cpu0;

    slam.save_frame_trajectory(out + "/frame_trajectory.txt", "TUM");
    slam.save_keyframe_trajectory(out + "/keyframe_trajectory.txt", "TUM");
    slam.shutdown();

    const int n = n_tracking + n_lost + n_init;
    std::ofstream js(out + "/stats.json");
    js << "{\n  \"frames\": " << n << ",\n  \"tracking\": " << n_tracking << ",\n  \"lost\": " << n_lost
       << ",\n  \"initializing\": " << n_init << ",\n  \"lost_events\": " << lost_events
       << ",\n  \"tracked_share\": " << (n ? static_cast<double>(n_tracking) / n : 0.0)
       << ",\n  \"track_ms_median\": " << percentile(track_ms, 0.5) << ",\n  \"track_ms_p95\": " << percentile(track_ms, 0.95)
       << ",\n  \"wall_s\": " << wall << ",\n  \"cpu_s\": " << cpu << ",\n  \"peak_rss_mb\": " << peak_rss_kb() / 1024.0
       << ",\n  \"loop_detector\": " << (flag(argc, argv, "--no-loop") ? "false" : "true") << "\n}\n";
    std::cout << "frames " << n << ", tracking " << n_tracking << ", lost " << n_lost << " (" << lost_events
              << " events), median " << percentile(track_ms, 0.5) << " ms, peak " << peak_rss_kb() / 1024.0 << " MB\n";
    return 0;
}
