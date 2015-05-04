#ifndef GL_ES
#define highp
#endif

uniform float iGlobalTime;
uniform vec3 iResolution;

void main() {
    highp vec2 position = gl_FragCoord.xy / iResolution.xy;

    highp float color = 0.0;
    color += sin(position.x * cos(iGlobalTime / 15.0) * 80.0) + cos(position.y * cos(iGlobalTime / 15.0) * 10.0);
    color += sin(position.y * sin(iGlobalTime / 10.0) * 40.0) + cos(position.x * sin(iGlobalTime / 25.0) * 40.0);
    color += sin(position.x * sin(iGlobalTime / 5.0)  * 10.0) + sin(position.y * sin(iGlobalTime / 35.0) * 80.0);
    color *= sin(iGlobalTime / 10.0) * 0.5;

    gl_FragColor = vec4(vec3(color, color * 0.5, sin(color + iGlobalTime / 3.0) * 0.75), 0.6);
}
