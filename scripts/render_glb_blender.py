"""Blender background renderer used for visual regression previews."""

from __future__ import annotations

import math
import os
import sys

import bpy
from mathutils import Vector


def look_at(camera, point: Vector) -> None:
    camera.rotation_euler = (point - camera.location).to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :]
    input_path, output_path = map(os.path.abspath, argv[:2])
    view_arg = argv[2] if len(argv) > 2 else "-1"
    solid = "solid" in argv[3:]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=input_path)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    corners = [
        obj.matrix_world @ Vector(corner)
        for obj in meshes
        for corner in obj.bound_box
    ]
    lower = Vector(
        (min(p.x for p in corners), min(p.y for p in corners), min(p.z for p in corners))
    )
    upper = Vector(
        (max(p.x for p in corners), max(p.y for p in corners), max(p.z for p in corners))
    )
    center = (lower + upper) * 0.5
    extent = upper - lower

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    distance = max(extent.x, extent.z) * 1.65
    axis_views = {
        "x+": Vector((1.0, 0.0, 0.02)),
        "x-": Vector((-1.0, 0.0, 0.02)),
        "y+": Vector((0.0, 1.0, 0.02)),
        "y-": Vector((0.0, -1.0, 0.02)),
    }
    if view_arg in axis_views:
        camera_offset = axis_views[view_arg] * distance
    else:
        view_sign = float(view_arg)
        camera_offset = Vector((0.0, view_sign * distance, 0.02 * extent.z))
    camera.location = center + camera_offset
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(extent.x, extent.z) * 1.08
    look_at(camera, center)

    key_data = bpy.data.lights.new("Key", "AREA")
    key_data.energy = 900
    key_data.shape = "DISK"
    key_data.size = max(extent) * 2.0
    key = bpy.data.objects.new("Key", key_data)
    bpy.context.collection.objects.link(key)
    view_direction = camera_offset.normalized()
    key.location = center + Vector(
        (-extent.x, view_direction.y * distance * 0.6, extent.z)
    )
    look_at(key, center)

    fill_data = bpy.data.lights.new("Fill", "AREA")
    fill_data.energy = 450
    fill_data.size = max(extent) * 2.0
    fill = bpy.data.objects.new("Fill", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = center + Vector(
        (extent.x, view_direction.y * distance * 0.35, 0.2 * extent.z)
    )
    look_at(fill, center)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH" if solid else "BLENDER_EEVEE"
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = output_path
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGBA"
    scene.world = bpy.data.worlds.new("World")
    scene.world.color = (0.055, 0.055, 0.055)
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.render.image_settings.color_depth = "8"
    if solid:
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "SINGLE"
        scene.display.shading.single_color = (0.35, 0.42, 0.5)
        scene.display.shading.show_shadows = True
        scene.display.shading.show_cavity = True
        scene.display.shading.cavity_type = "BOTH"

    for obj in meshes:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True

    bpy.ops.render.render(write_still=True)
    print(output_path)


if __name__ == "__main__":
    main()
