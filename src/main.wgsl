struct VertexInput {
    @location(0) pos: vec3f,
    @location(1) uv: vec2f
}

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f
}

@group(1) @binding(0)
var<uniform> camera: mat4x4f;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.pos = camera * vec4(in.pos, 1.0);
    out.uv = in.uv;
    return out;
}

@group(0) @binding(0)
var logo_t: texture_2d<f32>;
@group(0) @binding(1)
var logo_s: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return textureSample(logo_t, logo_s, in.uv);
}