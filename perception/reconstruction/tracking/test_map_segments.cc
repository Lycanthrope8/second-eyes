// Desktop test of map_segments.h (built and run by tests/test_map_segments.py when a C++ compiler is available).
#include "map_segments.h"
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>
using qtrack::MapSegmenter;

int main() {
    // 15 fps: 1 s initialising, 3 s tracking, 3 s lost (restart after 2 s), 2 s initialising, 2 s tracking.
    MapSegmenter s(2.0);
    std::vector<std::string> st;
    for (int k = 0; k < 15; ++k) st.push_back("Initializing");
    for (int k = 0; k < 45; ++k) st.push_back("Tracking");
    for (int k = 0; k < 45; ++k) st.push_back("Lost");
    for (int k = 0; k < 30; ++k) st.push_back("Initializing");
    for (int k = 0; k < 30; ++k) st.push_back("Tracking");
    int restarts = 0, restart_frame = -1;
    for (size_t i = 0; i < st.size(); ++i) {
        const double t = i / 15.0;
        if (s.step(t, st[i])) {
            ++restarts;
            restart_frame = static_cast<int>(i);
            s.on_restart();
        }
    }
    assert(restarts == 1);
    assert(restart_frame == 60 + 30);            // lost from frame 60 (4.0 s); 2.0 s later is frame 90
    assert(s.last_needs_saving());
    s.finish(true);
    const auto& m = s.maps();
    assert(m.size() == 2 && m[0].status == "saved" && m[1].status == "saved");
    assert(m[0].tracked_frames == 45 && m[1].tracked_frames == 30);
    assert(m[0].first_t > 0.99 && m[0].first_t < 1.01 && m[1].first_t > 8.99 && m[1].first_t < 9.01);

    // stella resets a young map by itself: Tracking then Initializing with no Lost in between
    MapSegmenter d(2.0);
    for (int k = 0; k < 10; ++k) d.step(k / 15.0, "Tracking");
    d.step(10 / 15.0, "Initializing");
    for (int k = 11; k < 40; ++k) d.step(k / 15.0, "Tracking");
    d.finish(true);
    assert(d.maps().size() == 2 && d.maps()[0].status == "discarded" && d.maps()[1].tracked_frames == 29);

    // restarts disabled, and a short loss that relocalises: one map
    MapSegmenter n(0.0);
    for (int k = 0; k < 20; ++k) n.step(k / 15.0, "Tracking");
    for (int k = 20; k < 80; ++k) assert(!n.step(k / 15.0, "Lost"));
    for (int k = 80; k < 90; ++k) n.step(k / 15.0, "Tracking");
    n.finish(true);
    assert(n.maps().size() == 1 && n.maps()[0].tracked_frames == 30);
    std::printf("map_segments: ok\n");
    return 0;
}
