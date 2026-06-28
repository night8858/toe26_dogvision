#include <dogvision_arm/arm_mission_protocol.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

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

    std::cout << "mission command tests passed" << std::endl;
    return 0;
}
