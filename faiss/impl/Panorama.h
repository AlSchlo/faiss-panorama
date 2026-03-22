/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_PANORAMA_H
#define FAISS_PANORAMA_H

#include <faiss/MetricType.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/PanoramaStats.h>
#include <faiss/impl/panorama_kernels/panorama_kernels.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace faiss {

/**
 * Implements the core logic of Panorama-based refinement.
 * arXiv: https://arxiv.org/abs/2510.00566
 *
 * Panorama partitions the dimensions of all vectors into L contiguous levels.
 * During the refinement stage of ANNS, it computes distances between the query
 * and its candidates level-by-level. After processing each level, it prunes the
 * candidates whose lower bound exceeds the k-th best distance.
 *
 * In order to enable speedups, the dimensions (or codes) of each vector are
 * stored in a batched, level-major manner. Within each batch of b vectors, the
 * dimensions corresponding to level 1 will be stored first (for all elements in
 * that batch), followed by level 2, and so on. This allows for efficient memory
 * access patterns.
 *
 * Coupled with the appropriate orthogonal PreTransform (e.g. PCA, Cayley,
 * etc.), Panorama can prune the vast majority of dimensions, greatly
 * accelerating the refinement stage.
 *
 * This is the abstract base class. Concrete subclasses (PanoramaFlat,
 * PanoramaPQ) implement compute_cumulative_sums and progressive_filter_batch
 * for their respective code formats.
 */
struct Panorama {
    static constexpr size_t kDefaultBatchSize = 128;

    size_t d = 0;
    size_t code_size = 0;
    size_t n_levels = 0;
    size_t level_width_bytes = 0;
    size_t batch_size = 0;

    Panorama() = default;
    Panorama(size_t d, size_t code_size, size_t n_levels, size_t batch_size);

    virtual ~Panorama() = default;

    void set_derived_values();

    /// Helper method to copy codes into level-oriented batch layout at a given
    /// offset in the list.
    /// PanoramaFlat uses row-major within each level (point bytes contiguous).
    /// PanoramaPQ overrides to use column-major (subquantizer columns
    /// contiguous).
    virtual void copy_codes_to_level_layout(
            uint8_t* codes,
            size_t offset,
            size_t n_entry,
            const uint8_t* code);

    /// Compute the cumulative sums (suffix norms) for database vectors.
    /// The cumsums follow the level-oriented batch layout to minimize the
    /// number of random memory accesses.
    /// Subclasses interpret the raw code bytes according to their format:
    /// PanoramaFlat reinterprets as float*, PanoramaPQ decodes via PQ.
    virtual void compute_cumulative_sums(
            float* cumsum_base,
            size_t offset,
            size_t n_entry,
            const uint8_t* code) const = 0;

    /// Compute the cumulative sums of the query vector.
    void compute_query_cum_sums(const float* query, float* query_cum_sums)
            const;

    /// Copy single entry (code and cum_sum) from one location to another.
    void copy_entry(
            uint8_t* dest_codes,
            uint8_t* src_codes,
            float* dest_cum_sums,
            float* src_cum_sums,
            size_t dest_idx,
            size_t src_idx) const;

    virtual void reconstruct(
            idx_t key,
            float* recons,
            const uint8_t* codes_base) const;
};

/**
 * Panorama for flat (uncompressed) float vectors.
 *
 * Codes are raw float vectors (code_size = d * sizeof(float)).
 * compute_cumulative_sums interprets codes as floats.
 * progressive_filter_batch computes dot products on raw float storage.
 *
 * When use_vertical_layout is true, codes are stored in column-major
 * order within each level (same as PanoramaPQ), and the scan uses
 * SIMD-ized compress + broadcast-FMA kernels from panorama_kernels.
 */
struct PanoramaFlat : Panorama {
    size_t level_width_dims = 0;
    bool use_vertical_layout = false;

    PanoramaFlat() = default;
    PanoramaFlat(size_t d, size_t n_levels, size_t batch_size);

    void copy_codes_to_level_layout(
            uint8_t* codes,
            size_t offset,
            size_t n_entry,
            const uint8_t* code) override;

    void reconstruct(idx_t key, float* recons, const uint8_t* codes_base)
            const override;

    void compute_cumulative_sums(
            float* cumsum_base,
            size_t offset,
            size_t n_entry,
            const uint8_t* code) const override;

    template <typename C, MetricType M>
    size_t progressive_filter_batch(
            const uint8_t* codes_base,
            const float* cum_sums,
            const float* query,
            const float* query_cum_sums,
            size_t batch_no,
            size_t list_size,
            const IDSelector* sel,
            const idx_t* ids,
            bool use_sel,
            std::vector<uint32_t>& active_indices,
            std::vector<float>& exact_distances,
            std::vector<uint8_t>& bitset,
            std::vector<float>& compressed_float_codes,
            float threshold,
            PanoramaStats& local_stats) const;
};

template <typename C, MetricType M>
size_t PanoramaFlat::progressive_filter_batch(
        const uint8_t* codes_base,
        const float* cum_sums,
        const float* query,
        const float* query_cum_sums,
        size_t batch_no,
        size_t list_size,
        const IDSelector* sel,
        const idx_t* ids,
        bool use_sel,
        std::vector<uint32_t>& active_indices,
        std::vector<float>& exact_distances,
        std::vector<uint8_t>& bitset,
        std::vector<float>& compressed_float_codes,
        float threshold,
        PanoramaStats& local_stats) const {
    const size_t bs = batch_size;
    size_t batch_start = batch_no * bs;
    size_t curr_batch_size = std::min(list_size - batch_start, bs);

    size_t cumsum_batch_offset = batch_no * bs * (n_levels + 1);
    const float* batch_cum_sums = cum_sums + cumsum_batch_offset;
    float q_norm = query_cum_sums[0] * query_cum_sums[0];

    size_t batch_offset_bytes = batch_no * bs * code_size;
    const uint8_t* storage_base = codes_base + batch_offset_bytes;

    if (use_vertical_layout) {
        // ----- Vertical (column-major) path: PQ-style kernels -----
        std::fill(bitset.begin(), bitset.end(), 0);
        size_t num_active = 0;
        size_t b_offset = batch_no * bs;

        for (size_t i = 0; i < curr_batch_size; i++) {
            size_t global_idx = batch_start + i;
            idx_t id = (ids == nullptr) ? global_idx : ids[global_idx];
            bool include = !use_sel || sel->is_member(id);
            if (!include)
                continue;

            active_indices[num_active] = b_offset + i;
            float cs = batch_cum_sums[i];
            if constexpr (M == METRIC_INNER_PRODUCT) {
                exact_distances[num_active] = 0.0f;
            } else {
                exact_distances[num_active] = cs * cs + q_norm;
            }
            bitset[i] = 1;
            num_active++;
        }

        if (num_active == 0) {
            return 0;
        }

        const float* batch_cums = cum_sums + b_offset * (n_levels + 1);
        const float* batch_codes_f =
                reinterpret_cast<const float*>(storage_base);

        size_t next_num_active = num_active;
        const size_t total_active = next_num_active;
        local_stats.total_dims += total_active * n_levels;

        constexpr float factor = (M == METRIC_INNER_PRODUCT) ? 1.0f : -2.0f;

        for (size_t level = 0; level < n_levels && next_num_active > 0;
             level++) {
            local_stats.total_dims_scanned += next_num_active;

            size_t actual_lwd =
                    std::min(level_width_dims, d - level * level_width_dims);

            float query_cum_norm;
            if constexpr (M == METRIC_INNER_PRODUCT) {
                query_cum_norm = -query_cum_sums[level + 1];
            } else {
                query_cum_norm = 2.0f * query_cum_sums[level + 1];
            }

            const float* cum_sums_level = batch_cums + bs * (level + 1);
            const float* codes_level =
                    batch_codes_f + bs * level_width_dims * level;
            const float* query_level = query + level * level_width_dims;

            bool is_sparse = next_num_active < bs / 16;

            size_t num_active_for_filtering = 0;
            if (is_sparse) {
                for (size_t di = 0; di < actual_lwd; di++) {
                    float q_val = factor * query_level[di];
                    const float* col = codes_level + di * bs;
                    for (size_t i = 0; i < next_num_active; i++) {
                        size_t real_idx = active_indices[i] - b_offset;
                        exact_distances[i] += q_val * col[real_idx];
                    }
                }
                num_active_for_filtering = next_num_active;
            } else {
                auto [cc, na] =
                        panorama_kernels::process_float_code_compression(
                                next_num_active,
                                bs,
                                actual_lwd,
                                compressed_float_codes.data(),
                                bitset.data(),
                                codes_level);

                panorama_kernels::process_float_level(
                        actual_lwd,
                        bs,
                        na,
                        query_level,
                        cc,
                        exact_distances.data(),
                        factor);
                num_active_for_filtering = na;
            }

            next_num_active = panorama_kernels::process_filtering(
                    num_active_for_filtering,
                    exact_distances.data(),
                    active_indices.data(),
                    const_cast<float*>(cum_sums_level),
                    bitset.data(),
                    b_offset,
                    0.0f,
                    query_cum_norm,
                    threshold);
        }

        return next_num_active;
    }

    // ----- Horizontal (row-major) path: original implementation -----
    const float* level_cum_sums = batch_cum_sums + bs;

    size_t num_active = 0;
    for (size_t i = 0; i < curr_batch_size; i++) {
        size_t global_idx = batch_start + i;
        idx_t id = (ids == nullptr) ? global_idx : ids[global_idx];
        bool include = !use_sel || sel->is_member(id);

        active_indices[num_active] = i;
        float cs = batch_cum_sums[i];

        if constexpr (M == METRIC_INNER_PRODUCT) {
            exact_distances[i] = 0.0f;
        } else {
            exact_distances[i] = cs * cs + q_norm;
        }

        num_active += include;
    }

    if (num_active == 0) {
        return 0;
    }

    size_t total_active = num_active;
    for (size_t level = 0; level < n_levels; level++) {
        local_stats.total_dims_scanned += num_active;
        local_stats.total_dims += total_active;

        float query_cum_norm = query_cum_sums[level + 1];

        size_t level_offset = level * level_width_bytes * bs;
        const float* level_storage =
                (const float*)(storage_base + level_offset);

        size_t next_active = 0;
        for (size_t i = 0; i < num_active; i++) {
            uint32_t idx = active_indices[i];
            size_t actual_level_width =
                    std::min(level_width_dims, d - level * level_width_dims);

            const float* yj = level_storage + idx * actual_level_width;
            const float* query_level = query + level * level_width_dims;

            float dot_product =
                    fvec_inner_product(query_level, yj, actual_level_width);

            if constexpr (M == METRIC_INNER_PRODUCT) {
                exact_distances[idx] += dot_product;
            } else {
                exact_distances[idx] -= 2.0f * dot_product;
            }

            float cum_sum = level_cum_sums[idx];
            float cauchy_schwarz_bound;
            if constexpr (M == METRIC_INNER_PRODUCT) {
                cauchy_schwarz_bound = -cum_sum * query_cum_norm;
            } else {
                cauchy_schwarz_bound = 2.0f * cum_sum * query_cum_norm;
            }

            float lower_bound = exact_distances[idx] - cauchy_schwarz_bound;

            active_indices[next_active] = idx;
            next_active += C::cmp(threshold, lower_bound) ? 1 : 0;
        }

        num_active = next_active;
        level_cum_sums += bs;
    }

    return num_active;
}
} // namespace faiss

#endif
