#version 330

uniform float u_bloom;

in vec2 v_uv;
in vec3 v_color;

out vec4 f_color;

void main() {
    // v_uv.x = normalized arc length (0 base -> 1 tip)
    // v_uv.y = across the ribbon width (0 -> 1)

    // Soft edge falloff across the ribbon. Without this a ribbon reads as a flat
    // strip; with it the strand looks round and catches light.
    float across = abs(v_uv.y - 0.5) * 2.0;
    float edge = 1.0 - smoothstep(0.55, 1.0, across);

    // Tip-to-base gradient. Spiderlily tepals pale toward the tip; the exponent
    // is tuned so the shift happens in the outer third only.
    float tipward = pow(v_uv.x, 2.2);
    vec3 tint = mix(v_color, mix(v_color, vec3(1.0, 0.88, 0.80), 0.5), tipward);

    // Bloom raises overall emission, so the plant brightens as it opens --
    // reinforcing the gesture-to-render coupling perceptually, not just spatially.
    float emissive = 0.72 + 0.60 * u_bloom + 0.32 * tipward;

    f_color = vec4(tint * emissive, edge);
}
