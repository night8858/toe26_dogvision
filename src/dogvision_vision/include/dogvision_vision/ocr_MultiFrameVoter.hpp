#pragma once

#include <cstddef>
#include <deque>
#include <optional>
#include <string>

/**
 * @brief 单帧 OCR 识别结果
 *
 * 包含识别出的数学表达式字符串及其计算结果。
 */
struct OCRVoteResult
{
    std::string expr;   ///< 识别出的数学表达式（如 "1+2=3"）
    int result = 0;     ///< 表达式的计算结果数值
    int mod4 = 0;       ///< result 对 4 取模的结果，用于方向分类
};

/**
 * @brief OCR 多帧投票器运行参数。
 *
 * 默认值保持旧版严格 3/3 行为；节点可从 settings.json 传入其他配置。
 */
struct OCRVoterConfig
{
    std::size_t window_size = 3;
    std::size_t min_occurrences = 3;
    double min_valid_ratio = 1.0;
    std::size_t lost_after_invalid_frames = 3;
};

/**
 * @brief 多帧投票器状态变更事件
 *
 * 标记稳定结果是否发生变化，供外部根据事件执行相应逻辑。
 */
enum class OCRVoteEvent
{
    None,           ///< 无事件，稳定结果未变化
    StableChanged,  ///< 产生了新的稳定结果（内容与之前不同）
    StableLost      ///< 连续无效帧达到配置阈值，稳定结果丢失
};

/**
 * @brief 多帧滑动窗口投票器，用于提高 OCR 识别结果的稳定性
 *
 * 将最近若干帧 OCR 结果存入可配置滑动窗口，并按最低票数和有效结果
 * 共识比例判定稳定表达式。同一稳定结果不会重复触发；票数并列时选择
 * 最近出现的表达式。
 *
 * 用法：
 * @code
 *   OCRMultiFrameVoter voter;
 *   auto event = voter.update(frame_result);
 *   if (event == OCRVoteEvent::StableChanged) {
 *       // 使用 voter.stable_result()
 *   }
 * @endcode
 */
class OCRMultiFrameVoter
{
public:
    /**
     * @brief 使用给定参数构造投票器。
     * @throws std::invalid_argument 参数范围或相互关系无效时抛出。
     */
    explicit OCRMultiFrameVoter(
        OCRVoterConfig config = OCRVoterConfig{});

    /**
     * @brief 输入一帧 OCR 结果，更新投票状态
     *
     * 将当前帧结果加入滑动窗口尾部；若窗口已满，移除最早的一帧。
     * 随后遍历窗口中的有效帧，统计各 expression 出现次数和占比。
     * 候选满足配置的最低票数和共识比例时，更新稳定结果。
     *
     * @param frame_result  当前帧的 OCR 识别结果。
     *                      传入 std::nullopt 表示该帧识别失败（无效帧）。
     * @return OCRVoteEvent 事件类型，调用方可根据返回值决定是否触发动作。
     *
     * @retval OCRVoteEvent::None           稳定结果未发生变化
     * @retval OCRVoteEvent::StableChanged  稳定结果更新为新表达式
     * @retval OCRVoteEvent::StableLost     稳定结果丢失
     */
    OCRVoteEvent update(const std::optional<OCRVoteResult>& frame_result);

    /**
     * @brief 重置投票器，清空历史窗口和稳定结果
     */
    void reset();

    /** @brief 当前是否有稳定结果 */
    bool has_stable_result() const;

    /**
     * @brief 获取当前稳定结果
     * @return 稳定结果的常引用
     * @throws std::logic_error 若当前无稳定结果
     */
    const OCRVoteResult& stable_result() const;

    /** @brief 当前窗口中已积累的帧数（最多为配置的 window_size） */
    std::size_t frame_count() const;

    /** @brief 当前窗口中有效（非 nullopt）帧的数量 */
    std::size_t valid_result_count() const;

private:
    OCRVoterConfig config_;

    /// 滑动窗口：存储最近 N 帧的 OCR 结果
    std::deque<std::optional<OCRVoteResult>> history_;

    /// 当前稳定结果（若有）
    std::optional<OCRVoteResult> stable_result_;

    /// 最近连续识别失败的帧数
    std::size_t consecutive_invalid_frames_ = 0;
};
