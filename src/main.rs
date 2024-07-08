use std::{
    cell::OnceCell,
    time::{Duration, Instant},
};

use glam::{Mat4, Quat, Vec3, Vec4};
use wgpu::{include_wgsl, util::DeviceExt};
use winit::{
    event::{DeviceEvent, ElementState, Event, KeyEvent, WindowEvent},
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
    window::CursorGrabMode,
};

#[pollster::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let event_loop = winit::event_loop::EventLoop::new()?;
    let window = winit::window::WindowBuilder::new().build(&event_loop)?;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let surface = instance.create_surface(&window)?;

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            power_preference: wgpu::PowerPreference::HighPerformance,
        })
        .await
        .ok_or("no satisfy adapter")?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::default(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await?;

    let cap = surface.get_capabilities(&adapter);
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: cap.formats[0],
        width: window.inner_size().width,
        height: window.inner_size().height,
        present_mode: cap.present_modes[0],
        alpha_mode: cap.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };

    surface.configure(&device, &config);

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(cube::VERTICES),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let indicens_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(cube::INDICENS),
        usage: wgpu::BufferUsages::INDEX,
    });

    let mut camera = Camera {
        pos: Vec3::new(0.0, 0.0, -3.0),
        forward: Quat::from_vec4(Vec4::Z),
    };

    let mut proj = Proj {
        fov: 60.0,
        aspect: config.width as f32 / config.height as f32,
        znear: 0.1,
        zfar: 100.0,
    };

    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[CameraUniform::new(&camera, &proj)]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let camera_binding_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let camera_binding_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &camera_binding_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
    });

    let gm_logo = include_bytes!("../gen-meta.jpeg");
    let gm_logo = image::load_from_memory(gm_logo)?.flipv();
    let gm_logo = gm_logo.to_rgba8();

    let texture_size = wgpu::Extent3d {
        width: gm_logo.width(),
        height: gm_logo.height(),
        depth_or_array_layers: 1,
    };

    let gm_logo_texture = device.create_texture_with_data(
        &queue,
        &wgpu::TextureDescriptor {
            label: None,
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &gm_logo,
    );

    let gm_logo_texture_view = gm_logo_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let gm_logo_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: None,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let texture_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

    let texture_binding_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &texture_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&gm_logo_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&gm_logo_sampler),
            },
        ],
    });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&texture_bind_group_layout, &camera_binding_group_layout],
        push_constant_ranges: &[],
    });

    let shader_module = device.create_shader_module(include_wgsl!("main.wgsl"));

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader_module,
            entry_point: "vs_main",
            buffers: &[Vertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_module,
            entry_point: "fs_main",
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: true,
        },
        multiview: None,
    });

    let mut time = OnceCell::new();
    let mut keyboard = Keyboard::new();

    window.set_cursor_grab(CursorGrabMode::Locked)?;
    window.set_cursor_visible(false);

    event_loop.run(|event, elwt| {
        let cur = Instant::now();
        let dt = time.get_or_init(|| cur).elapsed();

        match event {
            Event::WindowEvent { window_id, event } => match event {
                WindowEvent::Resized(size) => {
                    config.width = size.width;
                    config.height = size.height;
                    proj.aspect = config.width as f32 / config.height as f32;
                    surface.configure(&device, &config);
                }
                WindowEvent::RedrawRequested => {
                    *time.get_mut().unwrap() = cur;

                    camera.update_position(keyboard, dt);

                    queue.write_buffer(
                        &camera_buffer,
                        0,
                        bytemuck::cast_slice(&[CameraUniform::new(&camera, &proj)]),
                    );

                    let ouput = surface.get_current_texture().unwrap();
                    let view = ouput
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        ..Default::default()
                    });
                    render_pass.set_pipeline(&render_pipeline);
                    render_pass.set_bind_group(0, &texture_binding_group, &[]);
                    render_pass.set_bind_group(1, &camera_binding_group, &[]);
                    render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(indicens_buffer.slice(..), wgpu::IndexFormat::Uint16);
                    render_pass.draw_indexed(0..cube::INDICENS.len() as _, 0, 0..1);
                    drop(render_pass);

                    queue.submit(Some(encoder.finish()));
                    ouput.present();

                    window.request_redraw();
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if escape(&event) {
                        elwt.exit();
                    } else {
                        keyboard.handle_key_event(event);
                    }
                }
                WindowEvent::CloseRequested => {
                    elwt.exit();
                }
                _ => {}
            },
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                camera.handle_mouse_move((delta.0 as _, delta.1 as _), dt);
            }
            _ => {}
        }
    })?;
    Ok(())
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    uv: [f32; 2],
}

impl Vertex {
    const fn new(pos: [f32; 3], uv: [f32; 2]) -> Self {
        Self { pos, uv }
    }

    const fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: &[wgpu::VertexAttribute; 2] = &wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x2
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as _,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: ATTRIBUTES,
        }
    }
}

mod cube {
    use crate::Vertex;

    pub const VERTICES: &[Vertex] = &[
        // z-
        Vertex::new([-0.5, 0.5, 0.5], [0.0, 1.0]),  // A
        Vertex::new([0.5, 0.5, 0.5], [1.0, 1.0]),   // B
        Vertex::new([-0.5, -0.5, 0.5], [0.0, 0.0]), // C
        Vertex::new([0.5, -0.5, 0.5], [1.0, 0.0]),  // D
        // x+
        Vertex::new([0.5, 0.5, 0.5], [0.0, 1.0]),
        Vertex::new([0.5, 0.5, -0.5], [1.0, 1.0]),
        Vertex::new([0.5, -0.5, 0.5], [0.0, 0.0]),
        Vertex::new([0.5, -0.5, -0.5], [1.0, 0.0]),
        // z-
        Vertex::new([0.5, 0.5, -0.5], [0.0, 1.0]),
        Vertex::new([-0.5, 0.5, -0.5], [1.0, 1.0]),
        Vertex::new([0.5, -0.5, -0.5], [0.0, 0.0]),
        Vertex::new([-0.5, -0.5, -0.5], [1.0, 0.0]),
        // x-
        Vertex::new([-0.5, 0.5, -0.5], [0.0, 1.0]),
        Vertex::new([-0.5, 0.5, 0.5], [1.0, 1.0]),
        Vertex::new([-0.5, -0.5, -0.5], [0.0, 0.0]),
        Vertex::new([-0.5, -0.5, 0.5], [1.0, 0.0]),
        // y-
        Vertex::new([-0.5, -0.5, 0.5], [0.0, 1.0]),
        Vertex::new([0.5, -0.5, 0.5], [1.0, 1.0]),
        Vertex::new([-0.5, -0.5, -0.5], [0.0, 0.0]),
        Vertex::new([0.5, -0.5, -0.5], [1.0, 0.0]),
        // y+
        Vertex::new([-0.5, 0.5, -0.5], [0.0, 1.0]),
        Vertex::new([0.5, 0.5, -0.5], [1.0, 1.0]),
        Vertex::new([-0.5, 0.5, 0.5], [0.0, 0.0]),
        Vertex::new([0.5, 0.5, 0.5], [1.0, 0.0]),
    ];
    /// A B
    /// C D
    ///
    /// Ccw: ACB BCD
    pub const INDICENS: &[u16] = &[
        0, 2, 1, 1, 2, 3, //
        4, 6, 5, 5, 6, 7, //
        8, 10, 9, 9, 10, 11, //
        12, 14, 13, 13, 14, 15, //
        16, 18, 17, 17, 18, 19, //
        20, 22, 21, 21, 22, 23, //
    ];
}

#[derive(Debug, Clone, Copy)]
struct Proj {
    /// DEG
    fov: f32,
    aspect: f32,
    znear: f32,
    zfar: f32,
}
impl Proj {
    fn proj_mat(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov.to_radians(), self.aspect, self.znear, self.zfar)
    }
}

#[derive(Debug, Clone, Copy)]
struct Camera {
    pos: Vec3,
    forward: Quat,
}

#[derive(Default, Debug, Clone, Copy)]
struct Keyboard {
    w: bool,
    s: bool,
    a: bool,
    d: bool,
    dn: bool,
    up: bool,
}

impl Keyboard {
    fn new() -> Self {
        Self::default()
    }

    fn handle_key_event(&mut self, event: KeyEvent) {
        match event.physical_key {
            PhysicalKey::Code(KeyCode::KeyW) => self.w = event.state == ElementState::Pressed,
            PhysicalKey::Code(KeyCode::KeyS) => self.s = event.state == ElementState::Pressed,
            PhysicalKey::Code(KeyCode::KeyA) => self.a = event.state == ElementState::Pressed,
            PhysicalKey::Code(KeyCode::KeyD) => self.d = event.state == ElementState::Pressed,
            PhysicalKey::Code(KeyCode::Space) => self.up = event.state == ElementState::Pressed,
            PhysicalKey::Code(KeyCode::ShiftLeft) => self.dn = event.state == ElementState::Pressed,
            _ => {}
        }
    }
}

impl Camera {
    fn view_mat(&self) -> Mat4 {
        Mat4::look_at_rh(self.pos, self.pos + self.forward.mul_vec3(Vec3::Z), Vec3::Y)
    }

    fn update_position(&mut self, keyboard: Keyboard, dt: Duration) {
        let dt = dt.as_secs_f32();
        let mut d = Vec3::ZERO;

        if keyboard.w {
            d.z += dt;
        }
        if keyboard.s {
            d.z -= dt;
        }
        if keyboard.a {
            d.x -= dt;
        }
        if keyboard.d {
            d.x += dt;
        }
        if keyboard.up {
            d.y -= dt;
        }
        if keyboard.dn {
            d.y += dt;
        }

        self.pos += self.forward.mul_vec3(d);
    }

    fn handle_mouse_move(&mut self, (dx, dy): (f32, f32), dt: Duration) {
        let dt = dt.as_secs_f32() * 40.0;
        self.forward = Quat::from_rotation_y(-(dx * dt).to_radians())
            * self.forward
            * Quat::from_rotation_x(-(dy * dt).to_radians());
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform(Mat4);

impl CameraUniform {
    fn new(camera: &Camera, proj: &Proj) -> Self {
        Self(proj.proj_mat() * camera.view_mat() * Mat4::from_translation(-camera.pos))
    }
}

fn escape(event: &KeyEvent) -> bool {
    event.logical_key == Key::Named(NamedKey::Escape) && event.state == ElementState::Pressed
}
