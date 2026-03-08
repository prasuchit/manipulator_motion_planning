import argparse
import re

import mujoco


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_dir", required=True, help="Path to mujoco menagerie models"
    )
    parser.add_argument(
        "--output_path", default="../models/robots/ur5e_with_gripper.xml"
    )

    args = parser.parse_args()

    arm_spec = mujoco.MjSpec.from_file(
        f"{args.models_dir}/universal_robots_ur5e/ur5e.xml"
    )
    gripper_spec = mujoco.MjSpec.from_file(f"{args.models_dir}/robotiq_2f85/2f85.xml")

    attachment_site = next(s for s in arm_spec.sites if s.name == "attachment_site")
    arm_spec.attach(gripper_spec, site=attachment_site, prefix="gripper_", suffix="")

    xml = arm_spec.to_xml()

    # Strip keyframe block so adding scene objects doesn't break qpos size
    xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.DOTALL)

    with open(args.output_path, "w") as f:
        f.write(xml)

    print(f"Saved compose xml to {args.output_path}")


if __name__ == "__main__":
    main()
