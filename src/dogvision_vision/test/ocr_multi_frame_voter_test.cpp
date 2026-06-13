#include <dogvision_vision/ocr_MultiFrameVoter.hpp>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

namespace
{
OCRVoteResult result(const std::string& expr, int value = 0)
{
    return OCRVoteResult{expr, value, ((value % 4) + 4) % 4};
}

void expect(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << std::endl;
        std::exit(1);
    }
}

void add(OCRMultiFrameVoter& voter,
         const std::optional<OCRVoteResult>& value,
         OCRVoteEvent expected = OCRVoteEvent::None)
{
    expect(voter.update(value) == expected, "unexpected voter event");
}
} // namespace

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
