#version 330

uniform sampler2D u_source;
uniform bool u_horizontal;
uniform float u_threshold;
uniform vec2 u_texel;

in vec2 v_uv;
out vec4 f_color;

// 9-tap Gaussian using linear-filtering trick: 5 texture fetches cover 9 taps
// because sampling between texel centers averages two neighbours for free.
const float OFFSETS[5] = float[](0.0, 1.4117647, 3.2941176, 5.1764705, 7.0588234);
const float WEIGHTS[5] = float[](0.1963806, 0.2969070, 0.0944703, 0.0103813, 0.0002983);

vec3 bright(vec3 c) {
    if (u_threshold <= 0.0) return c;
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    // Soft knee rather than a hard cutoff: a hard threshold makes the glow pop
    // in and out as brightness crosses it, which is very visible in motion.
    float knee = smoothstep(u_threshold, u_threshold + 0.35, luma);
    return c * knee;
}

void main() {
    vec2 dir = u_horizontal ? vec2(u_texel.x, 0.0) : vec2(0.0, u_texel.y);
    vec3 acc = bright(texture(u_source, v_uv).rgb) * WEIGHTS[0];
    for (int i = 1; i < 5; ++i) {
        vec2 off = dir * OFFSETS[i];
        acc += bright(texture(u_source, v_uv + off).rgb) * WEIGHTS[i];
        acc += bright(texture(u_source, v_uv - off).rgb) * WEIGHTS[i];
    }
    f_color = vec4(acc, 1.0);
}
