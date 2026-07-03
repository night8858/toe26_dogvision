/**
 * @file ocr_multi_frame_voter_test.cpp
 * @brief OCR 多帧投票器单元测试
 *
 * 测试覆盖：
 *   - 3 次一致触发稳定结果
 *   - 2 次一致不触发
 *   - 2 同 1 异不触发
 *   - 同一稳定结果不重复触发
 *   - 新稳定结果替换旧结果
 *   - 3 帧无效后稳定结果丢失
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
        add(voter, result("1+1", 2));
        add(voter, result("1+1", 2));
        expect(!voter.has_stable_result(), "2/3 should not stabilize");
        add(voter, result("1+1", 2), OCRVoteEvent::StableChanged);
        expect(voter.stable_result().expr == "1+1", "3/3 should stabilize");
    }

    {
        OCRMultiFrameVoter voter;
        add(voter, result("2+2", 4));
        add(voter, result("2+2", 4));
        add(voter, result("3+3", 6));
        expect(!voter.has_stable_result(), "2 same plus 1 different must not stabilize");
        expect(voter.valid_result_count() == 3, "all valid frames should be counted");
    }

    {
        OCRMultiFrameVoter voter;
        add(voter, result("4+4", 8));
        add(voter, result("4+4", 8));
        add(voter, std::nullopt);
        expect(!voter.has_stable_result(), "2 valid plus invalid must not stabilize");
        expect(voter.valid_result_count() == 2, "invalid frames must not enter denominator");
    }

    {
        OCRMultiFrameVoter voter;
        add(voter, result("5+5", 10));
        add(voter, result("5+5", 10));
        add(voter, result("5+5", 10), OCRVoteEvent::StableChanged);
        add(voter, result("5+5", 10));
        expect(voter.stable_result().expr == "5+5",
               "same stable result should not repeat StableChanged");
    }

    {
        OCRMultiFrameVoter voter;
        add(voter, result("6+6", 12));
        add(voter, result("6+6", 12));
        add(voter, result("6+6", 12), OCRVoteEvent::StableChanged);
        add(voter, result("7+7", 14));
        add(voter, result("7+7", 14));
        expect(voter.stable_result().expr == "6+6", "2 new results must not replace old");
        add(voter, result("7+7", 14), OCRVoteEvent::StableChanged);
        expect(voter.stable_result().expr == "7+7", "new 3/3 stable target should replace old");
    }

    {
        OCRMultiFrameVoter voter;
        add(voter, result("8+8", 16));
        add(voter, result("8+8", 16));
        add(voter, result("8+8", 16), OCRVoteEvent::StableChanged);
        add(voter, std::nullopt);
        add(voter, std::nullopt);
        expect(voter.has_stable_result(), "stable result should survive 2 invalid frames");
        add(voter, std::nullopt, OCRVoteEvent::StableLost);
        expect(!voter.has_stable_result(), "3 invalid frames should clear stable result");
    }

    {
        OCRMultiFrameVoter voter;
        add(voter, result("9+9", 18));
        add(voter, result("9+9", 18));
        voter.reset();
        expect(!voter.has_stable_result(), "reset should clear stable result");
        expect(voter.frame_count() == 0, "reset should clear frame history");
    }

    std::cout << "OCRMultiFrameVoter tests passed" << std::endl;
    return 0;
}
