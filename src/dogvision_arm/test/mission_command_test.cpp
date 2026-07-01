#include <dogvision_arm/arm_mission_protocol.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

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

void expect_accept(dogvision_arm::ArmMissionController &controller,
                   const std::string &input,
                   const std::string &expected_low_level)
{
    const dogvision_arm::MissionCommandResult result =
        controller.handle_mission_command(input);
    expect(result.action == dogvision_arm::MissionCommandAction::Accept,
           "mission command should be accepted");
    expect(result.low_level == expected_low_level, "low-level command mismatch");
    expect(controller.busy(), "controller should be busy after accepting a command");

    const dogvision_arm::MissionStateResult ignored =
        controller.handle_arm_state("MODE:4DOF;L4:0;R4:0");
    expect(!ignored.completed, "non-DONE state must not complete the mission");
    expect(controller.busy(), "controller must stay busy before DONE");

    const dogvision_arm::MissionStateResult done = controller.handle_arm_state("DONE");
    expect(done.completed, "DONE should complete the mission");
    expect(done.feedback == "FEEDBACK:DONE", "DONE feedback mismatch");
    expect(done.completed_command == expected_low_level, "completed command mismatch");
    expect(!controller.busy(), "controller should be idle after DONE");
}

dogvision_arm::MissionStateResult force_timeout(dogvision_arm::ArmMissionController &controller)
{
    controller.set_timeout_ms(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    return controller.check_timeout();
}
} // namespace

int main()
{
    dogvision_arm::ArmMissionController controller;

    expect_accept(controller, "PICK,L,0.1,0.2,-0.3", "PICK,0,0.1,0.2,-0.3");
    expect_accept(controller, "PLACE,右,0.4,-0.5,-0.6", "PLACE,1,0.4,-0.5,-0.6");
    expect_accept(controller, "PUTBACK,LEFT", "PUTBACK,0");
    expect_accept(controller, "GETBACK,R", "GETBACK,1");
    expect_accept(controller, "PICKALL,0.1,0.2,0.3,0.4,0.5,0.6",
                  "PICKALL,0.1,0.2,0.3,0.4,0.5,0.6");
    expect_accept(controller, "PLACEALL,0.7,0.8,0.9,-0.1,-0.2,-0.3",
                  "PLACEALL,0.7,0.8,0.9,-0.1,-0.2,-0.3");
    expect_accept(controller, "PUTBACKALL", "PUTBACKALL");
    expect_accept(controller, "GETBACKALL", "GETBACKALL");

    const dogvision_arm::MissionCommandResult ignored =
        controller.handle_mission_command("FEEDBACK:DONE");
    expect(ignored.action == dogvision_arm::MissionCommandAction::Ignore,
           "feedback messages should be ignored");

    const dogvision_arm::MissionCommandResult invalid_place1 =
        controller.handle_mission_command("PLACE1,0,0.1,0.2,0.3");
    expect(invalid_place1.action == dogvision_arm::MissionCommandAction::Invalid,
           "PLACE1 must not be accepted");

    const dogvision_arm::MissionCommandResult invalid_place2 =
        controller.handle_mission_command("PLACE2,1,0.1,0.2,0.3");
    expect(invalid_place2.action == dogvision_arm::MissionCommandAction::Invalid,
           "PLACE2 must not be accepted");

    const dogvision_arm::MissionCommandResult active =
        controller.handle_mission_command("PICK,0,0.1,0.2,0.3");
    expect(active.action == dogvision_arm::MissionCommandAction::Accept,
           "busy test setup should accept first command");

    const dogvision_arm::MissionCommandResult busy =
        controller.handle_mission_command("GETBACKALL");
    expect(busy.action == dogvision_arm::MissionCommandAction::Busy,
           "new command should be rejected while busy");
    expect(busy.feedback == "FEEDBACK:BUSY", "busy feedback mismatch");
    expect(controller.active_command() == "PICK,0,0.1,0.2,0.3",
           "busy rejection must not overwrite active command");

    const dogvision_arm::MissionStateResult done = controller.handle_arm_state("DONE");
    expect(done.completed, "busy test cleanup should complete on DONE");

    dogvision_arm::ArmMissionController timeout_controller;
    const dogvision_arm::MissionCommandResult timeout_active =
        timeout_controller.handle_mission_command("PICK,0,0.1,0.2,0.3");
    expect(timeout_active.action == dogvision_arm::MissionCommandAction::Accept,
           "timeout test setup should accept first command");

    const dogvision_arm::MissionStateResult timeout = force_timeout(timeout_controller);
    expect(timeout.completed, "timeout should complete the wait state");
    expect(timeout.feedback == "FEEDBACK:TIMEOUT", "timeout feedback mismatch");
    expect(timeout.completed_command == "PICK,0,0.1,0.2,0.3",
           "timeout completed command mismatch");
    expect(!timeout_controller.busy(), "controller should release busy after timeout");

    const dogvision_arm::MissionStateResult late_done = timeout_controller.handle_arm_state("DONE");
    expect(late_done.completed, "late DONE should be accepted after timeout");
    expect(late_done.feedback == "FEEDBACK:DONE", "late DONE feedback mismatch");
    expect(late_done.completed_command == "PICK,0,0.1,0.2,0.3",
           "late DONE completed command mismatch");

    const dogvision_arm::MissionStateResult duplicate_done = timeout_controller.handle_arm_state("DONE");
    expect(!duplicate_done.completed, "late DONE should not be emitted twice");

    dogvision_arm::ArmMissionController replacement_controller;
    const dogvision_arm::MissionCommandResult first =
        replacement_controller.handle_mission_command("PICK,0,0.1,0.2,0.3");
    expect(first.action == dogvision_arm::MissionCommandAction::Accept,
           "replacement test setup should accept first command");
    const dogvision_arm::MissionStateResult first_timeout = force_timeout(replacement_controller);
    expect(first_timeout.completed, "replacement test first command should timeout");
    expect(first_timeout.feedback == "FEEDBACK:TIMEOUT",
           "replacement test timeout feedback mismatch");

    const dogvision_arm::MissionCommandResult second =
        replacement_controller.handle_mission_command("GETBACKALL");
    expect(second.action == dogvision_arm::MissionCommandAction::Accept,
           "new command should be accepted after timeout releases busy");

    const dogvision_arm::MissionStateResult second_done = replacement_controller.handle_arm_state("DONE");
    expect(second_done.completed, "new command should complete on DONE");
    expect(second_done.feedback == "FEEDBACK:DONE", "new command DONE feedback mismatch");
    expect(second_done.completed_command == "GETBACKALL",
           "new command must replace timed-out command for DONE feedback");

    std::cout << "mission command tests passed" << std::endl;
    return 0;
}
