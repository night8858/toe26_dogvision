#include <dogvision_arm/arm_internation.hpp>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <string>
#include <unistd.h>

namespace
{
void expect(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << std::endl;
        std::exit(1);
    }
}

uint8_t crc8(const uint8_t* data, size_t len)
{
    uint8_t crc = 0;
    for (size_t i = 0; i < len; ++i)
    {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
        {
            crc = (crc & 0x80) != 0
                ? static_cast<uint8_t>((crc << 1) ^ 0x07)
                : static_cast<uint8_t>(crc << 1);
        }
    }
    return crc;
}

std::array<uint8_t, 8> read_frame(int master_fd)
{
    std::array<uint8_t, 8> frame{};
    size_t received = 0;
    while (received < frame.size())
    {
        pollfd pfd{master_fd, POLLIN, 0};
        const int poll_result = ::poll(&pfd, 1, 1000);
        expect(poll_result > 0, "timed out waiting for answer frame");

        const ssize_t count =
            ::read(master_fd, frame.data() + received, frame.size() - received);
        if (count < 0 && errno == EINTR)
        {
            continue;
        }
        expect(count > 0, "failed to read answer frame");
        received += static_cast<size_t>(count);
    }
    return frame;
}
} // namespace

int main()
{
    const int master_fd = ::posix_openpt(O_RDWR | O_NOCTTY);
    expect(master_fd >= 0, "posix_openpt failed");
    expect(::grantpt(master_fd) == 0, "grantpt failed");
    expect(::unlockpt(master_fd) == 0, "unlockpt failed");

    const char* slave_name = ::ptsname(master_fd);
    expect(slave_name != nullptr, "ptsname failed");

    arm_internation arm;
    expect(arm.set_protocol_from_string("compiled"), "compiled protocol validation failed");
#if DOGVISION_ARM_USE_4DOF
    expect(arm.protocol() == ArmProtocol::Dof4BB, "expected compiled 4dof protocol");
    expect(arm.set_protocol_from_string("4dof"), "4dof alias should match compiled protocol");
    expect(!arm.set_protocol_from_string("aa"), "aa must not override compiled 4dof protocol");
    constexpr uint8_t expected_head = 0xBB;
#else
    expect(arm.protocol() == ArmProtocol::PlaneAA, "expected compiled aa protocol");
    expect(arm.set_protocol_from_string("aa"), "aa alias should match compiled protocol");
    expect(!arm.set_protocol_from_string("4dof"), "4dof must not override compiled aa protocol");
    constexpr uint8_t expected_head = 0xAA;
#endif
    expect(arm.open(slave_name, 115200), "failed to open pseudo serial port");

#if DOGVISION_ARM_USE_4DOF
    expect(!arm.send_arm_cmd(0, 1.0F, 2.0F), "AA command must fail in 4dof build");
#else
    expect(!arm.send_4dof_pose_cmd(0, 0.1F, 0.2F, 0.3F, 0.4F),
           "4dof command must fail in AA build");
#endif

    for (uint8_t answer = 0; answer <= 3; ++answer)
    {
        expect(arm.send_answer_cmd(answer), "send_answer_cmd failed");
        const std::array<uint8_t, 8> frame = read_frame(master_fd);
        const std::array<uint8_t, 7> expected_prefix = {
            expected_head, 0x05, answer, 0x00, 0x00, 0xFF, 0xEE
        };
        for (size_t i = 0; i < expected_prefix.size(); ++i)
        {
            expect(frame[i] == expected_prefix[i], "BB 05 frame payload mismatch");
        }
        expect(frame[7] == crc8(frame.data(), 7), "BB 05 CRC mismatch");
    }

    arm.close();
    ::close(master_fd);
    std::cout << arm.protocol_name() << " 05 answer frame tests passed" << std::endl;
    return 0;
}
