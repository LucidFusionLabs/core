
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 position = -1.0 + 2.0 * fragCoord.xy / iResolution.xy;

    float a = atan(position.y, position.x);
    float r = sqrt(dot(position, position));

    vec2 uv;
    uv.x = cos(a) / r;
    uv.y = sin(a) / r;
    uv /= 10.0;
    uv += iGlobalTime * 0.05;

    vec3 color = SampleChannelAtPoint(iChannel0, uv).rgb;
    fragColor = vec4(color * r * 1.5, 1.0);
}
