#include <dogvision_arm/arm_internation.hpp>

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <unistd.h>
#include <vector>

namespace
{
void expect(bool condition, const char *message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << std::endl;
        std::exit(1);
    }
}

uint8_t crc8(const uint8_t *data, size_t len)
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

std::vector<uint8_t> read_frame(int master_fd, size_t frame_len)
{
    std::vector<uint8_t> frame(frame_len, 0);
    size_t received = 0;
    while (received < frame.size())
    {
        pollfd pfd{master_fd, POLLIN, 0};
        const int poll_result = ::poll(&pfd, 1, 1000);
        expect(poll_result > 0, "timed out waiting for frame");

        const ssize_t count =
            ::read(master_fd, frame.data() + received, frame.size() - received);
        if (count < 0 && errno == EINTR)
        {
            continue;
        }
        expect(count > 0, "failed to read frame");
        received += static_cast<size_t>(count);
    }
    return frame;
}

float decode_float(const std::vector<uint8_t> &frame, size_t offset)
{
    float value = 0.0f;
    std::memcpy(&value, frame.data() + offset, sizeof(float));
    return value;
}

void expect_common_tail_crc(const std::vector<uint8_t> &frame)
{
    expect(frame.size() >= 5, "frame too short");
    expect(frame[0] == 0xBB, "frame head mismatch");
    expect(frame[frame.size() - 3] == 0xFF, "frame tail A mismatch");
    expect(frame[frame.size() - 2] == 0xEE, "frame tail B mismatch");
    expect(frame.back() == crc8(frame.data(), frame.size() - 1), "frame CRC mismatch");
}

void expect_float_near(float actual, float expected, const char *message)
{
    expect(std::fabs(actual - expected) < 1.0e-6f, message);
}

void expect_single_xyz_frame(const std::vector<uint8_t> &frame,
                             uint8_t cmd,
                             uint8_t arm_id,
                             float x,
                             float y,
                             float z)
{
    expect(frame.size() == 18, "single xyz frame length mismatch");
    expect_common_tail_crc(frame);
    expect(frame[1] == cmd, "single xyz cmd mismatch");
    expect(frame[2] == arm_id, "single xyz arm id mismatch");
    expect_float_near(decode_float(frame, 3), x, "single x mismatch");
    expect_float_near(decode_float(frame, 7), y, "single y mismatch");
    expect_float_near(decode_float(frame, 11), z, "single z mismatch");
}

void expect_single_back_frame(const std::vector<uint8_t> &frame, uint8_t cmd, uint8_t arm_id)
{
    expect(frame.size() == 6, "single back frame length mismatch");
    expect_common_tail_crc(frame);
    expect(frame[1] == cmd, "single back cmd mismatch");
    expect(frame[2] == arm_id, "single back arm id mismatch");
}

void expect_dual_xyz_frame(const std::vector<uint8_t> &frame,
                           uint8_t cmd,
                           float lx,
                           float ly,
                           float lz,
                           float rx,
                           float ry,
                           float rz)
{
    expect(frame.size() == 29, "dual xyz frame length mismatch");
    expect_common_tail_crc(frame);
    expect(frame[1] == cmd, "dual xyz cmd mismatch");
    expect_float_near(decode_float(frame, 2), lx, "dual lx mismatch");
    expect_float_near(decode_float(frame, 6), ly, "dual ly mismatch");
    expect_float_near(decode_float(frame, 10), lz, "dual lz mismatch");
    expect_float_near(decode_float(frame, 14), rx, "dual rx mismatch");
    expect_float_near(decode_float(frame, 18), ry, "dual ry mismatch");
    expect_float_near(decode_float(frame, 22), rz, "dual rz mismatch");
}

void expect_dual_back_frame(const std::vector<uint8_t> &frame, uint8_t cmd)
{
    expect(frame.size() == 5, "dual back frame length mismatch");
    expect_common_tail_crc(frame);
    expect(frame[1] == cmd, "dual back cmd mismatch");
}
} // namespace

int main()
{
    const int master_fd = ::posix_openpt(O_RDWR | O_NOCTTY);
    expect(master_fd >= 0, "posix_openpt failed");
    expect(::grantpt(master_fd) == 0, "grantpt failed");
    expect(::unlockpt(master_fd) == 0, "unlockpt failed");

    const char *slave_name = ::ptsname(master_fd);
    expect(slave_name != nullptr, "ptsname failed");

    arm_internation arm;
    expect(arm.protocol() == ArmProtocol::Dof4BB, "expected fixed 4dof protocol");
    expect(arm.set_protocol_from_string("compiled"), "compiled protocol validation failed");
    expect(arm.set_protocol_from_string("4dof"), "4dof alias should be accepted");
    expect(!arm.set_protocol_from_string("aa"), "aa alias must be rejected");
    expect(arm.open(slave_name, 115200), "failed to open pseudo serial port");

    expect(arm.send_4dof_pick_cmd(0, 0.45f, 0.42f, -0.21f), "send pick failed");
    expect_single_xyz_frame(read_frame(master_fd, 18), 0x11, 0, 0.45f, 0.42f, -0.21f);

    expect(arm.send_4dof_place_cmd(1, 0.46f, -0.41f, -0.20f), "send place failed");
    expect_single_xyz_frame(read_frame(master_fd, 18), 0x12, 1, 0.46f, -0.41f, -0.20f);

    expect(arm.send_4dof_put_block_back_cmd(0), "send putback failed");
    expect_single_back_frame(read_frame(master_fd, 6), 0x14, 0);

    expect(arm.send_4dof_get_block_back_cmd(1), "send getback failed");
    expect_single_back_frame(read_frame(master_fd, 6), 0x15, 1);

    expect(arm.send_4dof_pick_all_cmd(0.1f, 0.2f, 0.3f, 0.4f, -0.5f, 0.6f),
           "send pickall failed");
    expect_dual_xyz_frame(read_frame(master_fd, 29), 0x21, 0.1f, 0.2f, 0.3f, 0.4f, -0.5f, 0.6f);

    expect(arm.send_4dof_put_block_back_all_cmd(), "send putbackall failed");
    expect_dual_back_frame(read_frame(master_fd, 5), 0x22);

    expect(arm.send_4dof_place_all_cmd(0.7f, 0.8f, -0.9f, -0.1f, -0.2f, -0.3f),
           "send placeall failed");
    expect_dual_xyz_frame(read_frame(master_fd, 29), 0x23, 0.7f, 0.8f, -0.9f, -0.1f, -0.2f, -0.3f);

    expect(arm.send_4dof_get_block_back_all_cmd(), "send getbackall failed");
    expect_dual_back_frame(read_frame(master_fd, 5), 0x24);

    arm.close();
    ::close(master_fd);
    std::cout << "4DOF action frame tests passed" << std::endl;
    return 0;
}
