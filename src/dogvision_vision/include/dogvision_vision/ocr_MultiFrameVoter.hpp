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
 * @brief 多帧投票器状态变更事件
 *
 * 标记稳定结果是否发生变化，供外部根据事件执行相应逻辑。
 */
enum class OCRVoteEvent
{
    None,           ///< 无事件，稳定结果未变化
    StableChanged,  ///< 产生了新的稳定结果（内容与之前不同）
    StableLost      ///< 稳定结果丢失（窗口满且无有效帧）
};

/**
 * @brief 多帧滑动窗口投票器，用于提高 OCR 识别结果的稳定性
 *
 * 将连续多帧的 OCR 结果存入滑动窗口，统计各表达式出现的频次与占比，
 * 只有当某表达式同时满足：
 *   1. 出现次数 >= kMinOccurrences
 *   2. 有效帧中的占比 >= kMinValidRatio
 * 时才将其标记为"稳定结果"。
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
    /// 滑动窗口大小，最多保留最近的 N 帧数据
    static constexpr std::size_t kWindowSize = 10;

    /// 某表达式被认定为稳定所需的最低出现次数
    static constexpr std::size_t kMinOccurrences = 6;

    /// 某表达式在有效帧中所需的最低占比（0.0 ~ 1.0）
    static constexpr double kMinValidRatio = 0.60;

    /**
     * @brief 输入一帧 OCR 结果，更新投票状态
     *
     * 将当前帧结果加入滑动窗口尾部；若窗口已满，移除最早的一帧。
     * 随后遍历窗口中的有效帧，统计各 expression 出现次数和占比，
     * 若满足阈值条件则更新稳定结果。
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

    /** @brief 当前窗口中已积累的帧数（最多 kWindowSize） */
    std::size_t frame_count() const;

    /** @brief 当前窗口中有效（非 nullopt）帧的数量 */
    std::size_t valid_result_count() const;

private:
    /// 滑动窗口：存储最近 N 帧的 OCR 结果
    std::deque<std::optional<OCRVoteResult>> history_;

    /// 当前稳定结果（若有）
    std::optional<OCRVoteResult> stable_result_;
};
