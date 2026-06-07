# dogvision_arm

ROS2 Jazzy 机械臂控制包，包含串口通信节点、任务编排节点和终端命令节点。

```bash
ros2 launch dogvision_arm arm_control.launch
```

默认使用 `protocol:=4dof`，匹配 4DOF 双臂下位机的 `0xBB` 包头。旧 `0xAA` 平面机械臂协议需要显式指定：

```bash
ros2 launch dogvision_arm arm_control.launch protocol:=aa
```

调试终端：

```bash
ros2 run dogvision_arm arm_cmd_terminal_node
```

在终端输入 `help` 查看支持的高层任务命令和低层串口命令。

低层命令仍发布到 `/arm_internation/cmd`。旧 AA 平面臂命令保持不变：

```text
LF,X:10,Y:20
V,1,ON
P,ON,2500
```

4DOF 模式新增显式命令，避免旧 `LF/RF` 文本被误当成双臂控制：

```text
4POSE,L,X:0.1,Y:0.2,Z:0.3,PITCH:0.4
4POSE,R,0.1,0.2,0.3,0.4
4ACT,0
4ACT,1
```

4DOF 状态仍从 `/arm_internation/data` 发布。BB 模式下只发布 4DOF 双臂字段：

```text
MODE:4DOF;L4:x,y,z,pitch;R4:x,y,z,pitch;VALVE_BITS:n;MICRO_BITS:n
```
