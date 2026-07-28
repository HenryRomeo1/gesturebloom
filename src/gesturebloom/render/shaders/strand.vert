#version 330

uniform mat4 u_mvp;
uniform vec2 u_anchor;   // NDC offset so the plant can track a hand

in vec3 in_position;
in vec2 in_uv;
in vec3 in_color;

out vec2 v_uv;
out vec3 v_color;

void main() {
    v_uv = in_uv;
    v_color = in_color;
    vec4 clip = u_mvp * vec4(in_position, 1.0);
    // Applied after projection and scaled by w, so the shift is a constant
    // screen offset regardless of depth. Translating in world space instead
    // would make the plant shrink as it moved off the camera axis.
    clip.xy += u_anchor * clip.w;
    gl_Position = clip;
}
