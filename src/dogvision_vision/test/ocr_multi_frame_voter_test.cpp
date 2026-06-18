/**
 * @file ocr_multi_frame_voter_test.cpp
 * @brief OCR 多帧投票器单元测试
 *
 * 测试覆盖：
 *   - 正常稳定结果产生与替换
 *   - 无效帧不影响分母
 *   - 稳定结果丢失条件
 *   - A→B→A 模式的稳定性切换
 *   - reset 清空功能
 */

#include <dogvision_vision/ocr_MultiFrameVoter.hpp>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

namespace
{
/**
 * @brief 辅助函数：构建 OCRVoteResult
 */
OCRVoteResult result(const std::string& expr, int value = 0)
{
    return OCRVoteResult{expr, value, ((value % 4) + 4) % 4};
}

/**
 * @brief 断言条件为真，否则打印失败信息并退出
 */
void expect(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << std::endl;
        std::exit(1);
    }
}

/**
 * @brief 向投票器添加一帧并断言返回的事件类型
 */
void add(OCRMultiFrameVoter& voter,
         const std::optional<OCRVoteResult>& value,
         OCRVoteEvent expected = OCRVoteEvent::None)
{
    expect(voter.update(value) == expected, "unexpected voter event");
}
} // namespace

/**
 * @brief 测试入口：依次执行所有多帧投票测试用例
 */
int main()
{
    {
        OCRMultiFrameVoter voter;
        for (int i = 0; i < 5; ++i) add(voter, result("1+1", 2));
        for (int i = 0; i < 4; ++i) add(voter, result("2+2", 4));
        add(voter, result("1+1", 2), OCRVoteEvent::StableChanged);
        expect(voter.stable_result().expr == "1+1", "6/10 should stabilize");
    }

    {
        OCRMultiFrameVoter voter;
        for (int i = 0; i < 5; ++i) add(voter, result("3+3", 6));
        add(voter, result("4+4", 8));
        add(voter, result("3+3", 6), OCRVoteEvent::StableChanged);
        expect(voter.valid_result_count() == 7, "valid denominator should be 7");
    }

    {
        OCRMultiFrameVoter voter;
        for (int i = 0; i < 5; ++i) add(voter, result("5+5", 10));
        for (int i = 0; i < 5; ++i) add(voter, std::nullopt);
        expect(!voter.has_stable_result(), "5 occurrences must not stabilize");
    }

    {
        OCRMultiFrameVoter voter;
        for (int i = 0; i < 5; ++i) add(voter, result("6+6", 12));
        add(voter, result("6+6", 12), OCRVoteEvent::StableChanged);
        for (int i = 0; i < 4; ++i) add(voter, std::nullopt);
        expect(voter.has_stable_result(), "invalid frames must not enter ratio denominator");
        expect(voter.valid_result_count() == 6, "only valid results belong in denominator");
    }

    {
        OCRMultiFrameVoter voter;
        for (int i = 0; i < 5; ++i) add(voter, result("6+6", 12));
        add(voter, result("6+6", 12), OCRVoteEvent::StableChanged);
        for (int i = 0; i < 9; ++i) add(voter, std::nullopt);
        expect(voter.has_stable_result(), "stable result should survive 9 invalid frames");
        add(voter, std::nullopt, OCRVoteEvent::StableLost);
        expect(!voter.has_stable_result(), "10 invalid frames should clear stable result");
    }

    {
        OCRMultiFrameVoter voter;
        for (int i = 0; i < 5; ++i) add(voter, result("7+7", 14));
        add(voter, result("7+7", 14), OCRVoteEvent::StableChanged);
        for (int i = 0; i < 4; ++i) add(voter, result("8+8", 16));
        expect(voter.stable_result().expr == "7+7", "noise must not replace stable result");
        add(voter, result("8+8", 16));
        add(voter, result("8+8", 16), OCRVoteEvent::StableChanged);
        expect(voter.stable_result().expr == "8+8", "new stable target should replace old");

        for (int i = 0; i < 6; ++i)
        {
            const OCRVoteEvent expected =
                (i == 5) ? OCRVoteEvent::StableChanged : OCRVoteEvent::None;
            add(voter, result("7+7", 14), expected);
        }
        expect(voter.stable_result().expr == "7+7", "A-B-A should emit another change");

        voter.reset();
        expect(!voter.has_stable_result(), "reset should clear stable result");
        expect(voter.frame_count() == 0, "reset should clear frame history");
    }

    std::cout << "OCRMultiFrameVoter tests passed" << std::endl;
    return 0;
}
