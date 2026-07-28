#version 330

uniform mat4 u_mvp;

in vec3 in_position;
in vec2 in_uv;
in float in_role;

out vec2 v_uv;
out float v_role;

void main() {
    v_uv = in_uv;
    v_role = in_role;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
