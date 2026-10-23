"""Configure a session-local Gazebo SDF for same-axle leg collision."""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("--description-dir", default="/root/unitree_guide/src/unitree_ros/robots/TQBot2_description")
args = parser.parse_args()
source = Path(args.input).read_text().replace("ignition::corrected_offsets", "ignition_corrected_offsets")
tree = ET.ElementTree(ET.fromstring(source))
model = tree.getroot().find("model")
assert model is not None
for uri in model.iter("uri"):
    prefix = "model://TQBot2_description/"
    if uri.text and uri.text.startswith(prefix):
        uri.text = Path(args.description_dir, uri.text[len(prefix):]).as_uri()
# The sensor ROS plugin spins the global callback queue inside World::Update.
# A switch_controller service serviced there waits for the next World::Update
# and deadlocks. The model IMU plugin uses a dedicated callback queue.
for link in model.findall("link"):
    if link.get("name") == "base":
        sensor = ET.SubElement(link,"sensor",name="recover_ground_contact",type="contact")
        ET.SubElement(sensor,"always_on").text="true"
        ET.SubElement(sensor,"update_rate").text="100"
        contact = ET.SubElement(sensor,"contact")
        for collision in link.findall("collision"):
            ET.SubElement(contact,"collision").text=collision.get("name")
        plugin = ET.SubElement(sensor,"plugin",name="recover_ground_contact_ros",filename="libgazebo_ros_bumper.so")
        ET.SubElement(plugin,"bumperTopicName").text="/recover_ground_contacts"
        ET.SubElement(plugin,"frameName").text="world"
    for sensor in link.findall("sensor"):
        if sensor.get("type") == "imu":
            link.remove(sensor)
            plugin = ET.SubElement(model, "plugin", name="recover_imu", filename="libgazebo_ros_imu.so")
            for key, value in {"bodyName":link.get("name"), "topicName":"/trunk_imu", "frameName":"imu_link",
                               "updateRate":"1000", "gaussianNoise":"0", "xyzOffset":"0 0 0", "rpyOffset":"0 0 0"}.items():
                ET.SubElement(plugin,key).text = value
for link in model.findall("link"):
    name = link.get("name")
    group = 2 if name.startswith(("FR_", "FL_")) else 4 if name.startswith(("RR_", "RL_")) else 8
    self_collide = link.find("self_collide")
    if self_collide is None:
        self_collide = ET.SubElement(link,"self_collide")
    self_collide.text = "true" if group in (2,4) else "false"
    for collision in link.findall("collision"):
        surface = collision.find("surface")
        if surface is None:
            surface = ET.SubElement(collision,"surface")
        contact = surface.find("contact")
        if contact is None:
            contact = ET.SubElement(surface,"contact")
        mask = contact.find("collide_bitmask")
        if mask is None:
            mask = ET.SubElement(contact,"collide_bitmask")
        mask.text = str(group)
        # Fixed foot/knee collisions may have inherited their own self-collision
        # surface exclusion; this test uses the explicit axle mask instead.
        for flag in contact.findall("collide_without_contact"):
            contact.remove(flag)
tree.write(args.output,encoding="utf-8",xml_declaration=True)
print("Gazebo SDF: front mask=2, rear mask=4, body mask=8; connected links remain joint-filtered")
