#version 330

in vec2 in_position;
in vec2 in_uv;

out vec2 v_uv;

void main() {
    // Flip v: OpenCV numpy arrays have row 0 at the top, GL textures have it at
    // the bottom. Without this the camera feed renders upside down.
    v_uv = vec2(in_uv.x, 1.0 - in_uv.y);
    gl_Position = vec4(in_position, 0.0, 1.0);
}
