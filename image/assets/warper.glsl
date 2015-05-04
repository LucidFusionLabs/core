uniform float iGlobalTime;
uniform vec3 iResolution;
uniform sampler2D iChannel0;

void main(void) {
    vec2 position = -1.0 + 2.0 * gl_FragCoord.xy / iResolution.xy;

    float a = atan(position.y, position.x);
    float r = sqrt(dot(position, position));

    vec2 uv;
    uv.x = cos(a) / r;
    uv.y = sin(a) / r;
    uv /= 10.0;
    uv += iGlobalTime * 0.05;
    uv = mod(uv, 1.0);

    vec3 color = texture2D(iChannel0, uv).rgb;
    gl_FragColor = vec4(color * r * 1.5, 1.0);
}
