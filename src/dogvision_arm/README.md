# dogvision_arm

ROS2 Jazzy 机械臂控制包，包含串口通信节点、任务编排节点和终端命令节点。

```bash
ros2 launch dogvision_arm arm_control.launch
```

调试终端：

```bash
ros2 run dogvision_arm arm_cmd_terminal_node
```

在终端输入 `help` 查看支持的高层任务命令和低层串口命令。
