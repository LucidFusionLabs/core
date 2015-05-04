const float scale = 0.02;


vec2 getOffset(float time, vec2 uv)
{
  float a = 1.0 + 0.5 * sin(time + uv.x * 10.0);
  float b = 1.0 + 0.5 * cos(time + uv.y * 10.0);
	
  return scale * vec2(a + sin(b), b + cos(a));
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{

  float speed = 5.0;

  vec2 uv = fragCoord.xy / iResolution.xy;

  float time= speed * iGlobalTime;
  float prevTime= speed * (iGlobalTime-1.0);

  // current offset
  vec2 offset= getOffset(time, uv);	
	
  // offset at prev frame
  vec2 prevOffset= getOffset(prevTime, uv);	

  // motion vector from previous to current frame
  vec2 delta= offset - prevOffset;

  uv += offset;
	
  vec4 color= vec4(0.0, 0.0, 0.0, 0.0);
	
  // some iterations of unweighted blur
  const int steps= 20;
  float factor= 1.0 / float(steps);
  
  for (int i=0; i<steps; i++)
  {
     color += texture2D(iChannel0, uv);
	 uv += delta * factor;
  }
	
  vec4 whoaColor = color;
  float whoa = 0.1 + 0.01 * (1.0 + cos(10.0 * sin(time)));

  fragColor = (whoa * whoaColor) * color * factor;
}
