#version 330

uniform sampler2D u_camera;
uniform float u_dim;

in vec2 v_uv;
out vec4 f_color;

void main() {
    vec3 cam = texture(u_camera, v_uv).rgb;
    // Dimming the camera feed is what lets the flower read on top of it. At
    // u_dim = 1.0 the flower fights a bright, busy background and loses.
    f_color = vec4(cam * u_dim, 1.0);
}
