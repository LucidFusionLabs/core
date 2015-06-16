#ifndef GL_ES
#define lowp
#define highp
#endif

varying lowp vec4 DestinationColor;

#ifdef TEX2D
uniform bool TexCoordEnabled;
varying lowp vec2 TexCoordOut;
uniform lowp sampler2D iChannel0;
#endif

#ifdef TEXCUBE
uniform bool CubeMapEnabled;
varying lowp vec3 CubeCoordOut;
uniform lowp samplerCube CubeTexture;
#endif

void main(void) {
#ifdef TEXCUBE
    if (CubeMapEnabled) {
	    gl_FragColor = DestinationColor * textureCube(CubeTexture, CubeCoordOut);
    } else
#endif
    {
#ifdef TEX2D
	    gl_FragColor = TexCoordEnabled ? DestinationColor * texture2D(iChannel0, TexCoordOut) : DestinationColor; 
#else
	    gl_FragColor = DestinationColor; 
#endif
	}
    // gl_FragColor = pow(gl_FragColor, vec4(1.0/2.2)); // gamma correction
}
