#ifndef GL_ES
#define lowp
#define highp
#endif

uniform lowp float iGlobalTime;
uniform lowp vec3 iResolution;
uniform lowp sampler2D iChannel0;

void main(void) {
    lowp vec2 position = -1.0 + 2.0 * gl_FragCoord.xy / iResolution.xy;

    lowp float a = atan(position.y, position.x);
    lowp float r = sqrt(dot(position, position));

    lowp vec2 uv;
    uv.x = cos(a) / r;
    uv.y = sin(a) / r;
    uv /= 10.0;
    uv += iGlobalTime * 0.05;
    uv = mod(uv, 1.0);

    lowp vec3 color = texture2D(iChannel0, uv).rgb;
    gl_FragColor = vec4(color * r * 1.5, 1.0);
}
