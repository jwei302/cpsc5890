# Lab 1 — xArm7 Intro: Safety, GUI, FK/IK, Demos

## Objectives
- Set up an account on the lab computer
- Safety: turn the robot on/off, E-stop
- Use xArm Studio GUI to:
  - control the robot
  - record a demonstration
  - replay a demonstration
- Use code to compute FK and compare to GUI-reported EE pose
- IK: solve for joints given a target EE pose
- IK safety: collision detection / safe execution practices
- Collect a demo through GELLO (teleop)
- Replay that demo with joint-level position control
- Checkers: answer questions at the end (share github repo)

---

## 0. Ground rules (read before powering anything)
- Stay outside the robot’s workspace boundary; assume it may move unexpectedly.
- Do not “hot-plug” cables and do not put hands near joints when enabling motion.
- Keep a finger near E-stop during first motion tests.
- Use slow speeds for all first-time tests.

(See the xArm user manual safety section for general precautions and safe operating practices.) 

---

## 1. Account setup
1) Log in to the lab machine
2) Clone this repo into your home directory
3) Create / activate your Conda virtual python env
4) Install dependencies:
    ```bash
       pip install -r requirements.txt
    ```
Do **not** install packages system-wide.
    
---

## 2. Powering the Robot & E-Stop Drill (TF-guided)

A TA will first demonstrate:

- Control box power on/off
- Robot enable / disable
- Emergency stop (E-stop)
- Proper recovery after E-stop

Then each pair will:
- Power on the robot
- Enable motion
- Trigger the E-stop
- Recover safely

You may **not** proceed until a TF signs off.

---

## 3. GUI Warm-Up: Jog, Record, Replay

Open the **xArm GUI (xArm Studio)** and connect to the robot.

Tasks:
1. Jog the robot joints slowly
2. Jog the end-effector in Cartesian mode
3. Record a short trajectory
4. Replay the recorded trajectory

Answer the following:
- What coordinate frame is the TCP pose reported in?
- What are the units for position and orientation?

---

## 4. Code Warm-Up: Connect & Read State

Run the basic test script:

```bash
python scripts/00_basic_test.py --ip <ROBOT_IP>
```

This script should:
  - Connect to the robot
	-	Print joint angles
	-	Print TCP pose
	-	Exit cleanly

If this does not work, **do not move on** ask a TF.

---

## 5. Forward Kinematics vs GUI Pose

In this part, you will verify that your forward kinematics (FK) computation
matches the end-effector pose reported by the xArm GUI.

This step is critical for building intuition about:
- coordinate frames
- units (mm vs meters, degrees vs radians)
- what the robot controller considers the “true” pose

### Procedure

1. Move the robot to a **simple, static configuration**
   - Use a safe, reachable pose
   - Avoid joint limits and singularities

2. In the GUI:
   - Observe and record the TCP pose
   - Note the coordinate frame and units

3. Run the FK comparison script:

```bash
python scripts/01_fk_vs_gui_pose.py --ip <ROBOT_IP>
```

4.	The script will:
	-	Read the current joint angles from the robot
	-	Compute FK using the SDK
	-	Read the TCP pose reported by the controller
	-	Compute the difference between the two

### What to Record

For your lab submission, record:
	-	Joint angles used (in radians)
	-	GUI-reported TCP pose
	-	FK-computed TCP pose
	-	Position error (Euclidean norm in mm)
	-	Orientation error (Euclidean norm in radians)

Small numerical differences are expected due to floating-point precision.

### Discussion Questions
-	Why should FK computed by the controller match the GUI pose?
-	What might cause discrepancies between FK and GUI pose?
-	How could TCP offsets or calibration affect this comparison?

  ---

## 6. Inverse Kinematics & Safe Motion

In this section, you will compute inverse kinematics (IK) solutions for desired
end-effector poses and execute them **safely** on the robot.

This is your first time commanding autonomous motion from code.
Proceed slowly and deliberately.


### Objectives

By the end of this section, you should be able to:

- Specify a desired end-effector pose
- Compute an IK solution using the xArm SDK
- Execute the resulting joint motion safely
- Recognize when IK solutions are invalid or unsafe

### Running the IK Script

Use the provided script to compute IK and move the robot:

```bash
python scripts/02_ik_solve_and_move.py \
  --ip <ROBOT_IP> \
  --x 300 --y 0 --z 200 \
  --roll 3.14 --pitch 0 --yaw 0
```
The pose is specified as:
	- Position: (x, y, z) in millimeters
	-	Orientation: (roll, pitch, yaw) in radians

What the Script Should Do

Your implementation should:
	1.	Connect to the robot
	2.	Clear any existing warnings or errors
	3.	Enable safety features (collision detection, self-collision checking)
	4.	Compute an IK solution for the target pose
	5.	Move the robot using low joint speeds
	6.	Exit cleanly

If IK fails, the script should not move the robot.

Try several target poses and observe:
	•	Which poses produce valid IK solutions?
	•	Which poses fail (e.g., unreachable, singular)?
	•	How does the robot behave near joint limits?
	•	Does enabling collision detection change behavior?

##Discussion Questions
	- Why might a pose be geometrically reachable but unsafe?
	-	Why is it dangerous to test IK at high joint speeds?
	-	Why should faults be cleared before attempting motion?
	-	How could IK failures affect data collection in robot learning?

---

 ## 7. Record a Joint-Space Demonstration (Code)

In this section, you will record a robot demonstration **in joint space** using
the xArm Python SDK. This is a common way to collect data for robot learning,
debugging, and system identification.

### Recording a Demonstration

Run the following command:

```bash
python scripts/03_record_joint_traj.py \
  --ip <ROBOT_IP> \
  --out trajectories/demo1.json \
  --seconds 8
```

While recording:
	-	Move the robot slowly and smoothly
	-	Avoid abrupt changes in direction
	-	Stay away from joint limits and singular configurations
	-	Be ready to press the E-stop at all times

The script will save:
	-	Joint angles
	-	Timestamps

in a JSON file.

Inspecting the Recorded Trajectory

Open the saved trajectory file and check:
	-	Number of recorded waypoints
	-	Time spacing between waypoints
	-	Range of joint angles

Think about:
	-	How sampling rate affects smoothness
	-	How noise in demonstrations might affect learning
