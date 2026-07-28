#version 330

uniform sampler2D u_scene;
uniform sampler2D u_bloom_tex;
uniform float u_strength;

in vec2 v_uv;
out vec4 f_color;

void main() {
    vec3 scene = texture(u_scene, v_uv).rgb;
    vec3 glow = texture(u_bloom_tex, v_uv).rgb;
    vec3 hdr = scene + glow * u_strength;

    // ACES-ish filmic curve. Reinhard desaturates the hot core of the glow into
    // white; this keeps the red reading as red at high intensity.
    vec3 x = hdr * 0.6;
    vec3 mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    mapped = clamp(mapped, 0.0, 1.0);

    f_color = vec4(pow(mapped, vec3(1.0 / 2.2)), 1.0);
}
