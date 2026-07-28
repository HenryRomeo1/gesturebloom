#version 330

uniform mat4 u_mvp;
uniform vec2 u_anchor;   // NDC offset so the flower can grow from the hand

in vec3 in_position;
in vec2 in_uv;
in float in_role;

out vec2 v_uv;
out float v_role;

void main() {
    v_uv = in_uv;
    v_role = in_role;
    vec4 clip = u_mvp * vec4(in_position, 1.0);
    // Applied after projection, scaled by w so the shift is a constant screen
    // offset regardless of depth. Translating in world space instead would make
    // the flower shrink as it moves away from the camera axis.
    clip.xy += u_anchor * clip.w;
    gl_Position = clip;
}
