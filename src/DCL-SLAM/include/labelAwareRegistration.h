#ifndef _LABEL_AWARE_REGISTRATION_H_
#define _LABEL_AWARE_REGISTRATION_H_

#include <cmath>
#include <cstdint>
#include <unordered_map>

#include <pcl/correspondence.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/correspondence_rejection.h>

#include "paramsServer.h"

/**
 * Voxel-grid downsampler that preserves the *mode* (most frequent) semantic
 * label per voxel rather than averaging it.
 *
 * PCL's stock `VoxelGrid<PointPose3D>::setDownsampleAllData(true)` averages
 * every non-xyz field including `label`, which produces nonsense intermediate
 * label ids at class boundaries (e.g. road=1 + sidewalk=2 → 1.5 → 1 or 2).
 * For label-aware loop-closure verification we want label values to stay
 * canonical, so this routine picks the voxel's majority label instead.
 *
 * x, y, z, intensity are averaged (centroid) as normal.
 */
inline void voxelDownsampleModeLabel(
    const pcl::PointCloud<PointPose3D>& in,
    pcl::PointCloud<PointPose3D>& out,
    float leaf_size)
{
    out.clear();
    if (in.empty() || leaf_size <= 0.0f)
    {
        out = in;
        return;
    }

    struct Bucket
    {
        float sx = 0.0f, sy = 0.0f, sz = 0.0f;
        float sintensity = 0.0f;
        int count = 0;
        std::unordered_map<uint16_t, int> label_counts;
    };

    const float inv = 1.0f / leaf_size;
    std::unordered_map<int64_t, Bucket> buckets;
    buckets.reserve(in.size());

    auto key_of = [&](const PointPose3D& p) -> int64_t
    {
        const int32_t ix = static_cast<int32_t>(std::floor(p.x * inv));
        const int32_t iy = static_cast<int32_t>(std::floor(p.y * inv));
        const int32_t iz = static_cast<int32_t>(std::floor(p.z * inv));
        // Pack three 21-bit signed ints into one 64-bit key.
        return (static_cast<int64_t>(ix & 0x1FFFFF) << 42)
             | (static_cast<int64_t>(iy & 0x1FFFFF) << 21)
             |  static_cast<int64_t>(iz & 0x1FFFFF);
    };

    for (const auto& p : in.points)
    {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            continue;
        auto& b = buckets[key_of(p)];
        b.sx += p.x;
        b.sy += p.y;
        b.sz += p.z;
        b.sintensity += p.intensity;
        b.count++;
        b.label_counts[p.label]++;
    }

    out.points.reserve(buckets.size());
    for (auto& kv : buckets)
    {
        const auto& b = kv.second;
        PointPose3D q;
        const float invc = 1.0f / static_cast<float>(b.count);
        q.x = b.sx * invc;
        q.y = b.sy * invc;
        q.z = b.sz * invc;
        q.intensity = b.sintensity * invc;

        uint16_t mode_label = 0;
        int mode_count = -1;
        for (const auto& lc : b.label_counts)
        {
            if (lc.second > mode_count)
            {
                mode_count = lc.second;
                mode_label = lc.first;
            }
        }
        q.label = mode_label;
        out.points.push_back(q);
    }
    out.width = static_cast<uint32_t>(out.points.size());
    out.height = 1;
    out.is_dense = true;
}

/**
 * pcl::registration::CorrespondenceRejector that drops correspondence pairs
 * whose source.label differs from target.label.
 *
 * Usage: instantiate, call setInputSource()/setInputTarget() with the same
 * clouds you pass to ICP, then `icp.addCorrespondenceRejector(rej_shared_ptr)`.
 * For explicit (non-ICP) correspondence chains, call
 * `getRemainingCorrespondences(in, out)` directly.
 */
class LabelMatchRejector : public pcl::registration::CorrespondenceRejector
{
public:
    using Ptr = std::shared_ptr<LabelMatchRejector>;
    using CloudConstPtr = pcl::PointCloud<PointPose3D>::ConstPtr;

    LabelMatchRejector() { rejection_name_ = "LabelMatchRejector"; }

    void setInputSource(const CloudConstPtr& src) { source_ = src; }
    void setInputTarget(const CloudConstPtr& tgt) { target_ = tgt; }

    void getRemainingCorrespondences(
        const pcl::Correspondences& original,
        pcl::Correspondences& remaining) override
    {
        remaining.clear();
        if (!source_ || !target_)
        {
            remaining = original;
            return;
        }
        remaining.reserve(original.size());
        for (const auto& c : original)
        {
            if (c.index_query < 0 ||
                static_cast<size_t>(c.index_query) >= source_->size() ||
                c.index_match < 0 ||
                static_cast<size_t>(c.index_match) >= target_->size())
                continue;
            const uint16_t lq = source_->points[c.index_query].label;
            const uint16_t lm = target_->points[c.index_match].label;
            if (lq == lm)
                remaining.push_back(c);
        }
    }

protected:
    void applyRejection(pcl::Correspondences& correspondences) override
    {
        pcl::Correspondences tmp;
        getRemainingCorrespondences(correspondences, tmp);
        correspondences = std::move(tmp);
    }

private:
    CloudConstPtr source_;
    CloudConstPtr target_;
};

#endif  // _LABEL_AWARE_REGISTRATION_H_
