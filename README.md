# RelativityPathTracer
A real-time OpenCL path tracer to visualize the optical effects of Special Relativity

## Usage
`.\RelativityPathTracer.exe`
or
`.\RelativityPathTracer.exe < Scenes\[scene].txt`

## Controls
`w`, `a`, `s`, `d` to increase camera velocity in a given direction.  
`q` to add downward velocity to camera.  
`e` to add upward velocity to camera.  
`r` to reset camera's velocity.  
`[space]` to play/pause time. Scenes start paused by default.  
`i` to toggle light propagation. Scenes start with light propagation enabled by default, but default can be changed on a scene-by-scene basis.  

## Defining a Scene
When running without a scene file, custom objects may be added using the command line.  
The following syntax can be used on the command line or piped in from a scene file:  
1. `O[s|c|m#]`  
    Create a new object. `s` creates a sphere primitive, 
    `c` creates a cube primitive, 
    and `m#` creates a mesh, where `#` is the index of an imported mesh.  
    The following commands are applied to the most recently created object:  
    * `p#,#,#,#,#,#,#,#,#,#`  
      Set the translation, rotation, and scale of the object relative to the scene origin. Each `#` must be replaced by one floating-point value.
      The order of values is the following: `ptx,ty,tz,a,rx,ry,rz,sx,sy,sz` corresponds to a translation by `(tx, ty, tz)`, 
      a rotation by `a` radians about the axis `(rx, ry, rz)`, 
      and scaling by `sx` along the x-axis, `sy` along the y-axis, and `sz` along the z-axis.
    * `c#,#,#`  
      Set the color of the object. Each `#` must be replaced by one floating-point value.
      The color is stored in RGB Float where a value of `(1.0f, 1.0f, 1.0f)` is white.
      The brightness of the color will be affected by the scene's white-point (see `W#,#,#`).
      If the object is set to be a light (see `l#`), its color and brightness is determined by this color value.
      If the object is given a texture (see `t#`), this color value will not be used for the object's surface color.
    * `t#`  
      Assign a texture to the object. `#` must be replaced by a non-negative integer value corresponding to the index of the desired texture.
    * `l#`  
      If `#` is 0, the object will not emit light. 
      If `#` is non-zero, the object will become a light source whose brightness and color is determined by the object's color (see `c#,#,#`).
    * `v#,#,#`  
      Set the object's velocity. Each `#` must be replaced by a floating point value where `-1.0 < # < 1.0f`. 
      The order of values is `vx,y,z` for velocity vector `(x,y,z)`, whose length must be strictly less than `1.0`. 
      This velocity is represented in units of `c` (speed of light in a vacuum) and therefore an object's speed must be less than `1c`.
    * `f#,#`  
      Set the object to emit a periodic flash.
      Note: This flash does not emit light as if the object were a light source, instead it increases the brightness of the object's surface.
      The two values `#,#` correspond to two floating point values `frequency` and `duration` respectively.
      The object will increase in brightness once every `frequency` seconds, and will stay lit for `duration` seconds. 
      The object's proper time at each point on its surface is used to determine when to light up. 
      This means a moving object will appear to have a lower frequency (Time dilation), 
      and will also not appear to flash simultaneously across its surface (Relativity of simultaneity).
2. `T[filename]`  
   Import a new texture. Giving an invalid `filename` will cause the program to exit.
   The order in which textures are imported determines their "index", (see `t#`). The first texture imported has index `0`.
3. `M[filename]`  
   Import a new .obj mesh. Giving an invalid `filename`, including one that is not a `.obj` file, will cause the program to exit.
   The order in which meshes are imported determines their "index", (see `m#`). The first mesh imported has index `0`.
4. `A#`  
   Set the scene's ambient light intensity. `#` must be a floating point value.
   The ambient light determines how bright objects are when not lit by direct lighting.
   The appearance of ambiently lit objects is affected by the scene's white-point (see `W#,#,#`).
5. `W#,#,#`  
   Set the scene's white-point. This value determines the minimum intensity for a pixel to be fully lit. 
   Any pixels with greater intensity than the white point will be clamped to `1.0`, 
   and any pixels with less intensity will be scaled using [Hable tonemapping](http://filmicworlds.com/blog/filmic-tonemapping-operators/).
6. `I`  
   When the `I` command is given, propagation of light within the scene is set to be disabled by default. 
   A running scene can toggle this option by pressing the `i` key on the keyboard. With light propagation disabled,
   you see each point on a object as it exists in that instant, without accounting for how long it would take the light to reach your eyes.
   Light sources and shadows are disabled when light propagation is disabled.  Instead, ambient lighting is set to `1.0`.
   This is because there is no consistent way to trace the path of an infinite-speed ray.
7. `R`  
   Finalize scene generation, then run the application.
   If the `R` command is given at any point the scene file/command line will stop parsing further input.
   
## Scenes/Screenshots
1. `Scenes\arch.txt`  
    Stationary frame:  
   ![Stationary view](/Screenshots/arch1.png)  
   Camera moving towards the arch at 0.95c:  
   ![Moving view](/Screenshots/arch2.png)  
   Note that in the moving frame, it seems like the camera has not yet passed through the arch even though the back-side is visible.
2. `Scenes\bunny.txt`  
   Stationary frame:  
   ![Stationary view](/Screenshots/mesh1.png)  
   This shows off the ray tracer's ability to import and render complex meshes from .obj files
3. `Scenes\cube.txt`  
   Stationary frame:  
   ![Stationary view](/Screenshots/cube1.png)  
   Moving frame (0.9c to the right) without light propagation:  
   ![Moving view without light](/Screenshots/cube2.png)  
   Moving frame (0.9c to the right) with light propagation:  
   ![Moving view with light](/Screenshots/cube3.png)  
   With light propagation disabled, length contraction is clearly visible. 
   When enabled, the cube's apparent shape is changed significantly.
4. `Scenes\cubes.txt`  
   ![Cubes gif](/Screenshots/cubes.gif)  
   This scene shows two identical lines of cubes. One line is stationary with respect to the camera, the other is moving at 0.9c
5. `Scenes\ladder_paradox.txt`  
   This scene demonstrates the [Ladder Paradox](https://en.wikipedia.org/wiki/Ladder_paradox).
   In the stationary frame, a length-contracted ladder (represented by a meter stick) is able to fit inside a garage that is narrower than the ladder's rest length. The garage's front door is able to close, fully enclosing the length-contracted ladder. A few moments later, the garage's back door opens, allowing the ladder to continue moving.  
   ![Garage frame](/Screenshots/ladder_paradox_garage_frame.gif)  
   The paradox arises when one considers the ladder's rest frame. In this frame, the ladder is at its maximum length, and the garage is moving towards it at relativistic speeds. Due to length contraction, the garage will be even narrower than before, preventing the stationary ladder from fitting inside.  
   In reality, this situation is not paradoxical. Due to relativity of simultaneity, the order in which each door opens or closes can differ between reference frames. In the garage frame, the front door closes first, followed by the back door opening. In the ladder frame, as seen below, the back door must first open followed by the front door closing.
   ![Ladder frame](/Screenshots/ladder_paradox_ladder_frame.gif)  
6. `Scenes\rulers.txt`  
   This scene compares two identical rulers, both flashing on for 1 second followed by off for 1 second. The top ruler is moving to the right at 0.9c. The three main effects of special relativity can all be seen: length contraction, time dilation, and relativity of simultaneity.  
   ![Rulers](/Screenshots/rulers.gif)  
   Length contraction: The moving ruler is significantly shorter than its stationary counterpart.  
   Time dilation: The flashes on the moving ruler occur at a lower frequency.  
   Relativity of simultaneity: The flash occurs simultaneously across the stationary ruler. The moving ruler, however, sees the flash as occuring in a wave travelling from back to front.
7. `Scenes\shadows.txt`  
   ![Shadow 1](/Screenshots/shadows1.png)  
   ![Shadow 2](/Screenshots/shadows2.png)   
   ![Shadow 3](/Screenshots/shadows4.png)  
   ![Shadow 4](/Screenshots/shadows5.png)  
   Shadows take into account the propagation delay of light. As the light source moves through the scene, its shadows are delayed proportionally to their distance from the source.  
8. `Scenes\soccer.txt`  
   Stationary sphere:  
   ![Stationary sphere](/Screenshots/sphere_stationary.png)  
   Moving sphere:  
   ![Moving sphere](/Screenshots/sphere_moving.png)  
   This scene demonstrates the cancelling effect of light propagation on length contraction. Moving spheres, as predicted by Roger Penrose in his 1958 paper titled "The Apparent Shape of a Relativistically Moving Sphere", maintain their circular outline. 
## Credits
* Sam Lampere - [Ray Tracey's Blog: OpenCL path tracing tutorial](http://raytracey.blogspot.com/2016/11/opencl-path-tracing-tutorial-1-firing.html)  
  I used this tutorial as a guide to get the initial OpenCL ray tracing platform working. I have rewritten most of the OpenCL kernel,
  but a significant portion of the OpenCL-OpenGL interop setup code is still from this tutorial. I forked this project from [straaljager/OpenCL-path-tracing-tutorial-3-Part-1](https://github.com/straaljager/OpenCL-path-tracing-tutorial-3-Part-1).
* [Ray Tracing: Rendering a Triangle](https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection)  
  I used this article's code in the OpenCL kernel for ray-triangle intersection. It uses an algorithm found in "Fast, Minimum Storage Ray/Triangle Intersection by Möller & Trumbore"
* Tavian Barnes - [https://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/](https://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/)  
  I used this ray-box intersection algorithm in the OpenCL kernel.
* Majercik, Crassin, Shirley, McGuire - [A Ray-Box Intersection Algorithm and Efficient Dynamic Voxel Rendering](http://www.jcgt.org/published/0007/03/04/paper-lowres.pdf)  
  I also used this ray-box intersection algorithm in the OpenCL kernel.
