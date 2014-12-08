#ifndef GL_ES
#define highp
#endif

varying highp float TimeOut;
varying highp vec2 ResolutionOut;

void main() {
    highp vec2 position = gl_FragCoord.xy / ResolutionOut.xy;

    highp float color = 0.0;
    color += sin(position.x * cos(TimeOut / 15.0) * 80.0) + cos(position.y * cos(TimeOut / 15.0) * 10.0);
    color += sin(position.y * sin(TimeOut / 10.0) * 40.0) + cos(position.x * sin(TimeOut / 25.0) * 40.0);
    color += sin(position.x * sin(TimeOut / 5.0)  * 10.0) + sin(position.y * sin(TimeOut / 35.0) * 80.0);
    color *= sin(TimeOut / 10.0) * 0.5;

    gl_FragColor = vec4(vec3(color, color * 0.5, sin(color + TimeOut / 3.0) * 0.75), 0.6);
}
