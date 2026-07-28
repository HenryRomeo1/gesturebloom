#version 330

uniform sampler2D u_scene;
uniform sampler2D u_bloom_tex;
uniform sampler2D u_background;
uniform float u_strength;
uniform float u_has_background;

in vec2 v_uv;
out vec4 f_color;

void main() {
    vec3 scene = texture(u_scene, v_uv).rgb;
    vec3 glow = texture(u_bloom_tex, v_uv).rgb;
    vec3 hdr = scene + glow * u_strength;

    // ACES-ish filmic curve applied to the FLOWER ONLY. Tone mapping the camera
    // feed too would wash out the video; keeping them separate means the flower
    // gets filmic highlight rolloff while the camera stays true to what the
    // sensor saw.
    vec3 x = hdr * 0.6;
    vec3 mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    mapped = clamp(mapped, 0.0, 1.0);
    mapped = pow(mapped, vec3(1.0 / 2.2));

    // The camera background is already dimmed and gamma-correct, so add the
    // flower on top rather than blending -- emissive things add, they do not
    // occlude.
    vec3 bg = texture(u_background, v_uv).rgb * u_has_background;
    f_color = vec4(bg + mapped, 1.0);
}
