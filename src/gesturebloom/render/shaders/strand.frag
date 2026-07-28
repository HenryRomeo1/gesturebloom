#version 330

uniform vec3 u_tepal_color;
uniform vec3 u_stamen_color;
uniform float u_bloom;

in vec2 v_uv;
in float v_role;

out vec4 f_color;

void main() {
    // v_uv.x = normalized arc length (0 base -> 1 tip)
    // v_uv.y = across ribbon width (0 -> 1)

    // Soft edge falloff across the ribbon. Without this the ribbon reads as a
    // flat strip; with it the strand looks round and catches light.
    float across = abs(v_uv.y - 0.5) * 2.0;
    float edge = 1.0 - smoothstep(0.55, 1.0, across);

    // Tip-to-base gradient. Spiderlily tepals pale toward the tip; the mix
    // exponent is tuned so the shift happens in the outer third only.
    float tipward = pow(v_uv.x, 2.2);

    vec3 base = mix(u_tepal_color, u_stamen_color, v_role);
    vec3 tip = mix(base, vec3(1.0, 0.72, 0.62), 0.45);
    vec3 color = mix(base, tip, tipward);

    // Bloom raises overall emission so the flower brightens as it opens --
    // reinforcing the gesture->render coupling perceptually, not just spatially.
    float emissive = 0.75 + 0.65 * u_bloom + 0.35 * tipward;

    f_color = vec4(color * emissive, edge);
}
