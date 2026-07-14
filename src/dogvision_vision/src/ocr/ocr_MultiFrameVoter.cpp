#include <dogvision_vision/ocr_MultiFrameVoter.hpp>

#include <cmath>
#include <stdexcept>
#include <unordered_map>

OCRMultiFrameVoter::OCRMultiFrameVoter(OCRVoterConfig config)
    : config_(config)
{
    if (config_.window_size == 0)
        throw std::invalid_argument("OCR voter window_size must be > 0");
    if (config_.min_occurrences == 0 ||
        config_.min_occurrences > config_.window_size)
    {
        throw std::invalid_argument(
            "OCR voter min_occurrences must be in [1, window_size]");
    }
    if (!std::isfinite(config_.min_valid_ratio) ||
        config_.min_valid_ratio <= 0.0 ||
        config_.min_valid_ratio > 1.0)
    {
        throw std::invalid_argument(
            "OCR voter min_valid_ratio must be in (0, 1]");
    }
    if (config_.lost_after_invalid_frames == 0)
    {
        throw std::invalid_argument(
            "OCR voter lost_after_invalid_frames must be > 0");
    }
}

// ============================================================================
// 核心投票逻辑：输入一帧 OCR 结果，滑动窗口统计后返回事件
// ============================================================================
OCRVoteEvent OCRMultiFrameVoter::update(
    const std::optional<OCRVoteResult>& frame_result)
{
    // ---- 1. 将当前帧加入滑动窗口，保持窗口大小 ----
    history_.push_back(frame_result);
    if (history_.size() > config_.window_size)
    {
        history_.pop_front();
    }
    if (frame_result.has_value())
        consecutive_invalid_frames_ = 0;
    else
        ++consecutive_invalid_frames_;

    // 连续无效帧数独立于投票窗口配置，达到阈值时清空旧历史，避免
    // 清除稳定值后又被窗口中的旧票立即恢复。
    if (stable_result_.has_value() &&
        consecutive_invalid_frames_ >= config_.lost_after_invalid_frames)
    {
        history_.clear();
        stable_result_.reset();
        consecutive_invalid_frames_ = 0;
        return OCRVoteEvent::StableLost;
    }

    // ---- 2. 遍历窗口，统计各表达式出现的次数 ----
    std::unordered_map<std::string, std::size_t> counts;
    std::unordered_map<std::string, OCRVoteResult> results;
    std::unordered_map<std::string, std::size_t> latest_positions;
    std::size_t valid_count = 0;

    for (std::size_t index = 0; index < history_.size(); ++index)
    {
        const auto& item = history_[index];
        if (!item.has_value())
        {
            continue;  // 无效帧（识别失败），跳过
        }

        ++valid_count;
        ++counts[item->expr];
        results[item->expr] = *item;
        latest_positions[item->expr] = index;
    }

    // ---- 3. 找出现次数最多且满足阈值的候选表达式 ----
    const OCRVoteResult* candidate = nullptr;
    std::size_t candidate_count = 0;
    std::size_t candidate_latest_position = 0;
    for (const auto& entry : counts)
    {
        // 计算当前表达式在有效帧中的占比
        const double valid_ratio =
            static_cast<double>(entry.second) / static_cast<double>(valid_count);
        // 需同时满足：绝对次数下限 && 占比下限 && 比当前候选更优
        const std::size_t latest_position = latest_positions.at(entry.first);
        if (entry.second >= config_.min_occurrences &&
            valid_ratio >= config_.min_valid_ratio &&
            (entry.second > candidate_count ||
             (entry.second == candidate_count &&
              (candidate == nullptr ||
               latest_position > candidate_latest_position))))
        {
            candidate = &results.at(entry.first);
            candidate_count = entry.second;
            candidate_latest_position = latest_position;
        }
    }

    // ---- 4. 根据候选结果决定事件类型 ----
    if (candidate != nullptr)
    {
        // 存在合格候选：若与当前稳定结果不同，则更新并返回 StableChanged
        if (!stable_result_.has_value() || stable_result_->expr != candidate->expr)
        {
            stable_result_ = *candidate;
            return OCRVoteEvent::StableChanged;
        }
        // 与当前稳定结果相同，无需变更
        return OCRVoteEvent::None;
    }

    return OCRVoteEvent::None;
}

// ============================================================================
// 重置：清空所有历史数据和稳定结果
// ============================================================================
void OCRMultiFrameVoter::reset()
{
    history_.clear();
    stable_result_.reset();
    consecutive_invalid_frames_ = 0;
}

// ============================================================================
// 查询接口
// ============================================================================

bool OCRMultiFrameVoter::has_stable_result() const
{
    return stable_result_.has_value();
}

const OCRVoteResult& OCRMultiFrameVoter::stable_result() const
{
    if (!stable_result_.has_value())
    {
        throw std::logic_error("OCR voter has no stable result");
    }
    return *stable_result_;
}

std::size_t OCRMultiFrameVoter::frame_count() const
{
    return history_.size();
}

std::size_t OCRMultiFrameVoter::valid_result_count() const
{
    std::size_t count = 0;
    for (const auto& item : history_)
    {
        if (item.has_value())
        {
            ++count;
        }
    }
    return count;
}
