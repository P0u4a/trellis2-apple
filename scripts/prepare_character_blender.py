"""Create a cleaned Blender character scene with non-destructive donor assets."""

from __future__ import annotations

import os
import sys

import bmesh
import bpy


def move_to_collection(obj, collection):
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)


def remove_floor_component(obj) -> int:
    """Remove broad, flat disconnected components along the lowest Z plane."""
    if obj.type != "MESH":
        return 0
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    if not bm.faces:
        bm.free()
        return 0

    min_z = min(vertex.co.z for vertex in bm.verts)
    height_margin = 0.018
    seeds = {
        face
        for face in bm.faces
        if face.calc_center_median().z < min_z + height_margin
        and abs(face.normal.z) > 0.7
    }

    visited = set()
    floor_faces = set()
    for seed in seeds:
        if seed in visited:
            continue
        component = {seed}
        visited.add(seed)
        stack = [seed]
        component_verts = set(seed.verts)
        while stack:
            face = stack.pop()
            for edge in face.edges:
                for neighbor in edge.link_faces:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    component.add(neighbor)
                    component_verts.update(neighbor.verts)
                    stack.append(neighbor)
        zs = [vertex.co.z for vertex in component_verts]
        planar = max(zs) - min(zs) < 0.025
        xs = [vertex.co.x for vertex in component_verts]
        ys = [vertex.co.y for vertex in component_verts]
        center_x = (max(xs) + min(xs)) * 0.5
        center_y = (max(ys) + min(ys)) * 0.5
        compact = (max(xs) - min(xs) < 0.20) and (max(ys) - min(ys) < 0.26)
        under_shoe = (
            0.025 < abs(center_x) < 0.27
            and abs(center_y) < 0.22
            and compact
            and max(zs) - min(zs) >= 0.004
        )
        if planar and not under_shoe:
            floor_faces.update(component)

    removed = len(floor_faces)
    if floor_faces:
        bmesh.ops.delete(bm, geom=list(floor_faces), context="FACES")
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
        bm.to_mesh(obj.data)
        obj.data.update()
    bm.free()
    return removed


def import_glb(path):
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    return [obj for obj in bpy.context.scene.objects if obj not in before]


def create_precision_soles(base_objects, collection):
    """Cover TRELLIS' cropped shoe bottoms with small beveled sole meshes."""
    world_vertices = [
        obj.matrix_world @ vertex.co
        for obj in base_objects
        if obj.type == "MESH"
        for vertex in obj.data.vertices
    ]
    if not world_vertices:
        return []

    min_z = min(vertex.z for vertex in world_vertices)
    max_z = max(vertex.z for vertex in world_vertices)
    height = max_z - min_z
    sole_width = height * 0.090
    sole_length = height * 0.205
    sole_height = height * 0.020
    center_x = height * 0.095
    center_y = -height * 0.025

    material = bpy.data.materials.new("Precision_Sole_Black")
    material.diffuse_color = (0.0, 0.0, 0.0, 1.0)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    shader.inputs["Roughness"].default_value = 0.72
    if "Specular IOR Level" in shader.inputs:
        shader.inputs["Specular IOR Level"].default_value = 0.0

    soles = []
    for side, x in (("L", -center_x), ("R", center_x)):
        bpy.ops.mesh.primitive_cube_add(
            location=(x, center_y, min_z + sole_height * 0.52)
        )
        sole = bpy.context.object
        sole.name = f"Precision_Sole_{side}"
        sole.dimensions = (sole_width, sole_length, sole_height)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bevel = sole.modifiers.new(name="Rounded sole edge", type="BEVEL")
        bevel.width = sole_height * 0.28
        bevel.segments = 3
        bpy.context.view_layer.objects.active = sole
        bpy.ops.object.modifier_apply(modifier=bevel.name)
        sole.data.materials.append(material)
        for polygon in sole.data.polygons:
            polygon.use_smooth = True
        move_to_collection(sole, collection)
        soles.append(sole)
    return soles


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :]
    input_glb, output_glb, output_blend = map(os.path.abspath, argv[:3])
    donor_paths = [os.path.abspath(path) for path in argv[3:]]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    final_collection = bpy.data.collections.new("Final Character")
    bpy.context.scene.collection.children.link(final_collection)
    donor_collection = bpy.data.collections.new("Targeted Reconstruction Donors")
    bpy.context.scene.collection.children.link(donor_collection)

    base_objects = import_glb(input_glb)
    removed_faces = 0
    for obj in base_objects:
        move_to_collection(obj, final_collection)
        if obj.type == "MESH":
            obj.name = "Character_Final"
            removed_faces += remove_floor_component(obj)
            for polygon in obj.data.polygons:
                polygon.use_smooth = True
            for material in obj.data.materials:
                if material is not None:
                    material.name = f"{material.name}_Opaque"
    sole_objects = create_precision_soles(base_objects, final_collection)
    base_objects.extend(sole_objects)

    donor_names = ("Face_Donor_512", "Hand_Donor_512")
    for index, path in enumerate(donor_paths):
        imported = import_glb(path)
        for obj in imported:
            move_to_collection(obj, donor_collection)
            obj.name = donor_names[index] if index < len(donor_names) else f"Donor_{index}"
            obj.hide_render = True
            obj.hide_set(True)
    donor_collection.hide_render = True
    donor_collection.hide_viewport = True

    # Keep only the cleaned character selected for GLB export.
    bpy.ops.object.select_all(action="DESELECT")
    for obj in base_objects:
        obj.select_set(True)
    if base_objects:
        bpy.context.view_layer.objects.active = base_objects[0]
    bpy.ops.export_scene.gltf(
        filepath=output_glb,
        export_format="GLB",
        use_selection=True,
        export_apply=True,
    )

    bpy.context.scene["trellis_notes"] = (
        "Multi-view body with opaque hybrid PBR. Face donor is useful as a "
        "geometry reference; generated hand donors were rejected after validation. "
        "Generated floor removed and cropped shoe bottoms repaired with inset soles."
    )
    bpy.context.scene["floor_faces_removed"] = removed_faces
    bpy.ops.wm.save_as_mainfile(filepath=output_blend)
    print(f"Removed floor faces: {removed_faces:,}")
    print(output_glb)
    print(output_blend)


if __name__ == "__main__":
    main()
