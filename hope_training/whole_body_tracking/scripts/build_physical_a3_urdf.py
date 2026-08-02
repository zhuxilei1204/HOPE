"""Build a physical-evaluation A3 URDF with one non-overlapping blade collider."""

from __future__ import annotations

import argparse
import pathlib
import xml.etree.ElementTree as ET


def set_tiny_inertial(
    link: ET.Element,
    *,
    mass_value: str,
    inertia_value: str,
) -> None:
    inertial = link.find("inertial")
    if inertial is None:
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    mass = inertial.find("mass")
    if mass is None:
        mass = ET.SubElement(inertial, "mass")
    inertia = inertial.find("inertia")
    if inertia is None:
        inertia = ET.SubElement(inertial, "inertia")
    mass.set("value", mass_value)
    for name in ("ixx", "iyy", "izz"):
        inertia.set(name, inertia_value)
    for name in ("ixy", "ixz", "iyz"):
        inertia.set(name, "0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--remove-collision-link",
        action="append",
        default=[],
        help="Link whose collision geometry should be removed; repeatable.",
    )
    parser.add_argument("--blade-link", default="pingpang_red_Link")
    return parser.parse_args()


def build_physical_urdf(
    source: pathlib.Path,
    output: pathlib.Path,
    *,
    remove_collision_links: tuple[str, ...],
    blade_link: str,
) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    links_by_name = {
        link.get("name"): link for link in root.findall("link")
    }
    for link_name in remove_collision_links:
        link = links_by_name.get(link_name)
        if link is None:
            raise ValueError(f"missing collision-removal link {link_name!r}")
        collisions = list(link.findall("collision"))
        if len(collisions) != 1:
            raise ValueError(
                f"expected exactly one collision on {link_name!r}, "
                f"found {len(collisions)}"
            )
        link.remove(collisions[0])
        if link.find("visual") is None:
            raise ValueError(f"{link_name!r} must retain its visual geometry")
        inertial = link.find("inertial")
        mass = inertial.find("mass") if inertial is not None else None
        if mass is None or float(mass.get("value", "0")) <= 0.0:
            set_tiny_inertial(
                link,
                mass_value="0.000001",
                inertia_value="1e-9",
            )

    blade = links_by_name.get(blade_link)
    if blade is None or len(blade.findall("collision")) != 1:
        raise ValueError(f"blade link {blade_link!r} must retain one collision")
    set_tiny_inertial(
        blade,
        mass_value="0.0001",
        inertia_value="1e-7",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)

    parsed = ET.parse(output).getroot()
    generated_links = {
        link.get("name"): link for link in parsed.findall("link")
    }
    for link_name in remove_collision_links:
        if generated_links[link_name].find("collision") is not None:
            raise RuntimeError(
                f"generated physical URDF still contains {link_name!r} collision"
            )
    if len(generated_links[blade_link].findall("collision")) != 1:
        raise RuntimeError("generated physical URDF lost its blade collision")


def main() -> int:
    args = parse_args()
    remove_collision_links = tuple(args.remove_collision_link) or (
        "right_hand_pingpang_Link",
        "pingpang_black_Link",
        "pingbang_ball_Link",
    )
    build_physical_urdf(
        args.source.resolve(),
        args.output.resolve(),
        remove_collision_links=remove_collision_links,
        blade_link=args.blade_link,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
