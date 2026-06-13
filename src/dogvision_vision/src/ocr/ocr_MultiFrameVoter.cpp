#include <dogvision_vision/ocr_MultiFrameVoter.hpp>

#include <stdexcept>
#include <unordered_map>

// ============================================================================
// 核心投票逻辑：输入一帧 OCR 结果，滑动窗口统计后返回事件
// ============================================================================
OCRVoteEvent OCRMultiFrameVoter::update(
    const std::optional<OCRVoteResult>& frame_result)
{
    // ---- 1. 将当前帧加入滑动窗口，保持窗口大小 ----
    history_.push_back(frame_result);
    if (history_.size() > kWindowSize)
    {
        history_.pop_front();
    }

    // ---- 2. 遍历窗口，统计各表达式出现的次数 ----
    std::unordered_map<std::string, std::size_t> counts;
    std::unordered_map<std::string, OCRVoteResult> results;
    std::size_t valid_count = 0;

    for (const auto& item : history_)
    {
        if (!item.has_value())
        {
            continue;  // 无效帧（识别失败），跳过
        }

        ++valid_count;
        ++counts[item->expr];
        results[item->expr] = *item;
    }

    // ---- 3. 找出现次数最多且满足阈值的候选表达式 ----
    const OCRVoteResult* candidate = nullptr;
    std::size_t candidate_count = 0;
    for (const auto& entry : counts)
    {
        // 计算当前表达式在有效帧中的占比
        const double valid_ratio =
            static_cast<double>(entry.second) / static_cast<double>(valid_count);
        // 需同时满足：绝对次数下限 && 占比下限 && 比当前候选更优
        if (entry.second >= kMinOccurrences &&
            valid_ratio >= kMinValidRatio &&
            entry.second > candidate_count)
        {
            candidate = &results.at(entry.first);
            candidate_count = entry.second;
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

    // 无合格候选：
    // 若窗口已满且连续无有效帧，则认为稳定结果已丢失
    if (stable_result_.has_value() &&
        history_.size() == kWindowSize &&
        valid_count == 0)
    {
        stable_result_.reset();
        return OCRVoteEvent::StableLost;
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
