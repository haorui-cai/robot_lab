"""Simulation-only Recover actor bridge and repeated supine evaluation."""
import argparse
import csv
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState, ModelStates, ContactsState
from gazebo_msgs.srv import SetModelConfiguration, SetModelState
from sensor_msgs.msg import JointState, Imu
from std_srvs.srv import Empty, Trigger
from controller_manager_msgs.srv import SwitchController
from unitree_legged_msgs.msg import MotorCmd
from geometry_msgs.msg import WrenchStamped

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True)
parser.add_argument("--run-dir", required=True)
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--observe-seconds", type=float, default=17.)
args = parser.parse_args()
run = Path(args.run_dir)
run.mkdir(parents=True, exist_ok=True)
rospy.init_node("TQBot2_recover_test")
names = [f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL") for joint in ("hip", "thigh", "calf")]
default = np.tile([0., 1.36, -2.65], 4)
scale = np.tile([1.588, 4.75, 1.735], 4)
low = np.tile([-1.588, -3.141, -2.975], 4)
high = np.tile([1.588, 6.108, -.916], 4)
weights = np.load(args.policy)
layers = len(weights.files)//2
state = {"q": None, "dq": None, "imu": None, "pose": None, "twist": None}
contacts = {}
description = ET.fromstring(rospy.get_param("/robot_description"))
trunk_offset = np.array([float(v) for v in description.find("joint[@name='floating_base']/origin").get("xyz").split()])

def contact_cb(msg, name):
    contacts[name] = (msg.header.stamp.to_sec(), float(np.linalg.norm([msg.wrench.force.x,msg.wrench.force.y,msg.wrench.force.z])))

for name in ["body_shell"] + [leg+"_"+part for leg in ("FR","FL","RR","RL") for part in ("foot","knee")]:
    rospy.Subscriber("/visual/"+name+"_contact/the_force", WrenchStamped, contact_cb, callback_args=name, queue_size=1)

def bumper_cb(msg):
    force = 0.
    for contact in msg.states:
        if "ground_plane" in contact.collision1_name or "ground_plane" in contact.collision2_name:
            f = contact.total_wrench.force
            force += float(np.linalg.norm([f.x,f.y,f.z]))
    contacts["ground_bumper"] = (msg.header.stamp.to_sec(),force)

rospy.Subscriber("/recover_ground_contacts", ContactsState, bumper_cb, queue_size=1)

def motion():
    twist = state["twist"]
    return (np.linalg.norm([twist.linear.x,twist.linear.y,twist.linear.z]),
            np.linalg.norm([twist.angular.x,twist.angular.y,twist.angular.z]))

def ground_force(now):
    return sum(force for stamp,force in contacts.values() if 0 <= now-stamp < .1)

def joint_cb(msg):
    if not all(n in msg.name for n in names):
        return
    index = [msg.name.index(n) for n in names]
    state["q"] = np.array(msg.position)[index]
    state["dq"] = np.array(msg.velocity)[index]

def model_cb(msg):
    if "TQBot2_gazebo" in msg.name:
        i = msg.name.index("TQBot2_gazebo")
        state["pose"], state["twist"] = msg.pose[i], msg.twist[i]

rospy.Subscriber("/TQBot2_gazebo/joint_states", JointState, joint_cb, queue_size=1)
rospy.Subscriber("/trunk_imu", Imu, lambda msg: state.update(imu=msg), queue_size=1)
rospy.Subscriber("/gazebo/model_states", ModelStates, model_cb, queue_size=1)
pubs = [rospy.Publisher(f"/TQBot2_gazebo/{n[:-6]}_controller/command", MotorCmd, queue_size=1) for n in names]
for service in ("pause_physics", "unpause_physics", "set_model_configuration", "set_model_state"):
    rospy.wait_for_service("/gazebo/"+service, timeout=60)
pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
set_q = rospy.ServiceProxy("/gazebo/set_model_configuration", SetModelConfiguration)
set_pose = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
switch = rospy.ServiceProxy("/TQBot2_gazebo/controller_manager/switch_controller", SwitchController)
reset_history = rospy.ServiceProxy("/TQBot2_gazebo/reset_hardware_history", Trigger)
controllers = [n[:-6]+"_controller" for n in names]

def rotation(q):
    x,y,z,w = q.x,q.y,q.z,q.w
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def command(target):
    for i, publisher in enumerate(pubs):
        msg = MotorCmd()
        msg.mode, msg.q, msg.dq, msg.tau = 10, float(target[i]), 0., 0.
        msg.Kp, msg.Kd = (60.,3.) if i%3 == 2 else (40.,2.)
        publisher.publish(msg)

unpause()
deadline = time.monotonic()+30
while any(state[k] is None for k in state) or any(p.get_num_connections()==0 for p in pubs):
    if time.monotonic()>deadline:
        raise RuntimeError("Missing observations: %s; missing controller connections: %s" %
                           ([k for k in state if state[k] is None], [names[i] for i,p in enumerate(pubs) if p.get_num_connections()==0]))
    time.sleep(.05)
results = []
with (run/"trajectory.csv").open("w") as output:
    writer = csv.writer(output)
    writer.writerow(["episode", "time", "gravity_z", "trunk_height", "linear_speed", "angular_speed", "ground_force", "max_joint_speed", "target_change"])
    for episode in range(args.episodes):
        assert switch([], controllers, 2, False, 5.).ok
        pause()
        yaw = episode*2*np.pi/args.episodes
        model = ModelState()
        model.model_name, model.reference_frame = "TQBot2_gazebo", "world"
        model.pose.position.z = .096 + trunk_offset[2]
        model.pose.orientation.x, model.pose.orientation.y = np.cos(yaw/2), np.sin(yaw/2)
        response = set_q("TQBot2_gazebo", "robot_description", names, default.tolist())
        assert response.success, response.status_message
        response = set_pose(model)
        assert response.success, response.status_message
        assert reset_history().success
        previous = np.zeros(12)
        unpause()
        assert switch(controllers, [], 2, False, 5.).ok
        # Hold the reset pose; actor inference is forbidden before landing.
        landing_start = rospy.Time.now().to_sec()
        stable_since = None
        last_landing_log = -1
        rate = rospy.Rate(50)
        print("Waiting for ground contact and one second of settled motion", flush=True)
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            command(default)
            linear,angular = motion()
            landed = ground_force(now)>30 and linear<.08 and angular<.5
            if int(now-landing_start) != last_landing_log:
                last_landing_log = int(now-landing_start)
                print("Landing %.2fs: contact %.1f N, speed %.3f m/s, angular %.3f rad/s" %
                      (now-landing_start,ground_force(now),linear,angular),flush=True)
            stable_since = (now if stable_since is None else stable_since) if landed else None
            if stable_since is not None and now-stable_since >= 1.:
                break
            if now-landing_start > 10.:
                pause()
                raise RuntimeError("Robot did not settle on ground; recovery actor was not started")
            rate.sleep()
        landing_seconds = now-landing_start
        start_ground_force = ground_force(now)
        initial_gravity_z = float((rotation(state["imu"].orientation).T @ np.array([0.,0.,-1.]))[2])
        assert initial_gravity_z > .99, "Gazebo reset must start feet-up: %s" % initial_gravity_z
        print("Starting grounded feet-up episode %d, yaw %.1f degrees after %.3f s; contact %.1f N" % (episode, np.degrees(yaw),landing_seconds,start_ground_force), flush=True)
        start, last, streak, recovery = rospy.Time.now().to_sec(), -1., 0, None
        rate = rospy.Rate(50)
        previous_target = default.copy()
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            elapsed = now-start
            if elapsed >= args.observe_seconds:
                break
            if now <= last:
                rate.sleep()
                continue
            last = now
            imu = state["imu"]
            r = rotation(imu.orientation)
            gravity = r.T @ np.array([0.,0.,-1.])
            gyro = np.array([imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z])
            # Noetic gazebo_ros_imu.cpp rotates world angular velocity with R
            # instead of inverse R; undo that rotation, then express in body.
            # https://github.com/ros-simulation/gazebo_ros_pkgs/blob/noetic-devel/gazebo_plugins/src/gazebo_ros_imu.cpp
            gyro = r.T @ r.T @ gyro
            obs = np.concatenate([gravity, np.clip(gyro,-20,20), state["q"]-default,
                                  np.clip(state["dq"],-30,30)*.1, previous]).astype(np.float32)
            action = obs
            for i in range(layers):
                action = weights[f"w{i}"] @ action + weights[f"b{i}"]
                if i < layers-1:
                    action = np.where(action>0,action,np.expm1(np.minimum(action,0)))
            assert np.isfinite(action).all()
            previous = np.clip(action,-1,1)
            target = np.clip(default+scale*previous,low,high)
            target_change = float(np.max(np.abs(target-previous_target)))
            command(target)
            previous_target = target.copy()
            pose, twist = state["pose"],state["twist"]
            height = pose.position.z + (rotation(pose.orientation) @ trunk_offset)[2]
            linear = np.linalg.norm([twist.linear.x,twist.linear.y,twist.linear.z])
            angular = np.linalg.norm([twist.angular.x,twist.angular.y,twist.angular.z])
            good = gravity[2] < -.95 and .02 < height < .18 and linear < .15 and angular < .5 and elapsed >= 1
            streak = streak+1 if good else 0
            writer.writerow([episode,elapsed,gravity[2],height,linear,angular,ground_force(now),float(np.max(np.abs(state["dq"]))),target_change])
            if streak >= 25 and recovery is None:
                recovery = elapsed
            rate.sleep()
        result = {"episode":episode,"yaw":float(yaw),"initial_gravity_z":initial_gravity_z,
                  "landing_wait_seconds":landing_seconds,"actor_start_contact_force":start_ground_force,
                  "numeric_body_stability":recovery is not None,"numeric_body_stability_seconds":recovery,
                  "final_gravity_z":float(gravity[2]),"final_height":float(height),
                  "final_linear_speed":float(linear),"final_angular_speed":float(angular)}
        results.append(result)
        print(json.dumps(result),flush=True)
    pause()
    report = {"episodes":results,"requested_recovery_success_rate":None,
              "observation_source":"IMU and joint states; Gazebo model state used only for evaluation height/speeds",
              "support_contact_verified":True,
              "note":"Body-axis stability is not requested prone recovery; inspect legs and final video. Actor remains active for the full observation period to expose post-flip jitter."}
    (run/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
