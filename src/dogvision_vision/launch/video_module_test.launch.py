"""Replay one video through YOLO, PPOCR, or both visual test modules."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _shutdown_on_replay_failure(event, context):
    del context
    if event.returncode != 0:
        return [EmitEvent(event=Shutdown(reason="video replay process failed"))]
    return None


def _launch_setup(context):
    target = LaunchConfiguration("target").perform(context).strip().lower()
    if target not in {"yolo", "ppocr", "all"}:
        raise RuntimeError("target must be one of: yolo, ppocr, all")

    common_topic = LaunchConfiguration("image_topic")
    eof_topic = LaunchConfiguration("eof_topic")
    config_path = LaunchConfiguration("config_path")
    save_video = LaunchConfiguration("save_video")
    show_window = LaunchConfiguration("show_window")
    nodes = []

    if target in {"yolo", "all"}:
        nodes.append(
            Node(
                package="dogvision_vision",
                executable="yolo_accuracy_test_node",
                name="yolo_video_test_node",
                output="screen",
                parameters=[
                    {
                        "config_path": config_path,
                        "image_source": "topic",
                        "image_topic": common_topic,
                        "eof_topic": eof_topic,
                        "show_window": show_window,
                        "save_video": save_video,
                        "output_dir": LaunchConfiguration("yolo_output_dir"),
                        "video_fps": LaunchConfiguration("yolo_output_fps"),
                        "enable_undistort": LaunchConfiguration("enable_undistort"),
                    }
                ],
            )
        )

    if target in {"ppocr", "all"}:
        nodes.append(
            Node(
                package="dogvision_vision",
                executable="ppocr_node",
                name="ppocr_video_test_node",
                output="screen",
                parameters=[
                    {
                        "config_path": config_path,
                        "mode": "test",
                        "image_source": "topic",
                        "image_topic": common_topic,
                        "eof_topic": eof_topic,
                        "dynamic_image_subscription": False,
                        "enable_keyboard_trigger": False,
                        "show_visual": show_window,
                        "show_ocr_roi": False,
                        "show_debug_panels": False,
                        "save_video": save_video,
                        "save_result_images": LaunchConfiguration(
                            "save_result_images"
                        ),
                        "yaml_path": LaunchConfiguration("ocr_yaml_path"),
                        "debug_snapshot_dir": LaunchConfiguration("ocr_debug_dir"),
                    }
                ],
            )
        )

    replay = Node(
        package="dogvision_vision",
        executable="video_replay_node",
        name="video_replay_node",
        output="screen",
        parameters=[
            {
                "input_path": LaunchConfiguration("input_video"),
                "image_topic": common_topic,
                "eof_topic": eof_topic,
                "frame_id": "video",
                "required_subscribers": 2 if target == "all" else 1,
            }
        ],
    )
    nodes.append(replay)
    nodes.append(
        RegisterEventHandler(
            OnProcessExit(
                target_action=replay,
                on_exit=_shutdown_on_replay_failure,
            )
        )
    )
    return nodes


def generate_launch_description():
    share = FindPackageShare("dogvision_vision")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "input_video", description="Absolute path to the input video"
            ),
            DeclareLaunchArgument(
                "target", default_value="all", description="yolo, ppocr, or all"
            ),
            DeclareLaunchArgument(
                "config_path",
                default_value=PathJoinSubstitution([share, "config", "settings.json"]),
            ),
            DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("eof_topic", default_value="/video_replay/eof"),
            DeclareLaunchArgument("show_window", default_value="false"),
            DeclareLaunchArgument("save_video", default_value="true"),
            DeclareLaunchArgument("save_result_images", default_value="true"),
            DeclareLaunchArgument("enable_undistort", default_value="true"),
            DeclareLaunchArgument("yolo_output_fps", default_value="30.0"),
            DeclareLaunchArgument(
                "yolo_output_dir",
                default_value=PathJoinSubstitution([share, "data", "yolotest"]),
            ),
            DeclareLaunchArgument(
                "ocr_yaml_path",
                default_value=PathJoinSubstitution(
                    [share, "data", "ocr_output", "ocr_results.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "ocr_debug_dir",
                default_value=PathJoinSubstitution([share, "data", "ocr_debug"]),
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
