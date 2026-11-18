// map_segments.h: which map each frame belongs to, and when to stop waiting for relocalisation and start a new map.
//
// stella_vslam keeps one map. When tracking is lost it tries to relocalise on every frame, which only works if the
// camera comes back to a place that map already covers; on a walk through a house it usually doesn't. qtrack therefore
// saves the map and starts a new one once tracking has been lost for a set time (source-video seconds). stella also
// resets a map by itself when tracking is lost within a few seconds of starting it; such a map is "discarded" and its
// frames have no saved pose.
#pragma once
#include <string>
#include <vector>

namespace qtrack {

struct MapRecord {
    int id = 0;
    std::string status = "live";        // live, saved, discarded (stella reset it soon after it started), empty
    double first_t = -1.0, last_t = -1.0;   // first and last frames reported Tracking in this map
    long tracked_frames = 0;
};

class MapSegmenter {
public:
    explicit MapSegmenter(double restart_after_s) : restart_after_(restart_after_s) { maps_.push_back(MapRecord{}); }

    // Call once per frame with the tracker's state after that frame. Returns true when the current map should be saved
    // and a new one started: the caller saves the map's trajectory, requests stella's reset, then calls on_restart().
    bool step(double t, const std::string& state) {
        MapRecord& m = maps_.back();
        if (state == "Tracking") {
            if (m.first_t < 0) m.first_t = t;
            m.last_t = t;
            ++m.tracked_frames;
            lost_since_ = -1.0;
            has_map_ = true;
            return false;
        }
        if (state == "Lost") {
            if (lost_since_ < 0) lost_since_ = t;
            return restart_after_ > 0 && has_map_ && t - lost_since_ >= restart_after_;
        }
        // Initializing after a map existed: stella reset that map by itself
        if (has_map_) {
            const int next = m.id + 1;
            m.status = "discarded";
            MapRecord n;
            n.id = next;
            maps_.push_back(n);
            has_map_ = false;
        }
        lost_since_ = -1.0;
        return false;
    }

    void on_restart() {   // the current map was saved; the frames that follow start the next one
        maps_.back().status = "saved";
        MapRecord n;
        n.id = maps_.back().id + 1;
        maps_.push_back(n);
        has_map_ = false;
        lost_since_ = -1.0;
    }

    // At the end: whether the last map still needs saving.
    bool last_needs_saving() const { return maps_.back().status == "live" && maps_.back().tracked_frames > 0; }
    void finish(bool saved) {
        MapRecord& m = maps_.back();
        if (m.status == "live") m.status = saved ? "saved" : (m.tracked_frames > 0 ? "live" : "empty");
    }
    int current_id() const { return maps_.back().id; }
    const std::vector<MapRecord>& maps() const { return maps_; }

private:
    double restart_after_;
    double lost_since_ = -1.0;
    bool has_map_ = false;
    std::vector<MapRecord> maps_;
};

}  // namespace qtrack
