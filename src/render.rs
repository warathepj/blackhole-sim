use eframe::wgpu;
use eframe::wgpu::util::DeviceExt;
use nalgebra::{Point3, Vector3 as NVec3};

use crate::physics::SimulationState;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LensingUniforms {
    pub camera_pos: [f32; 3],
    pub bh_mass: f32,
    pub camera_forward: [f32; 3],
    pub rs: f32,
    pub camera_right: [f32; 3],
    pub time: f32,
    pub camera_up: [f32; 3],
    pub screen_width: f32,
    pub screen_height: f32,
    pub _pad: [f32; 3], // padding to align total size to multiple of 16 (80 bytes)
}

pub struct Camera {
    pub theta: f32,  // Azimuthal angle
    pub phi: f32,    // Polar angle
    pub radius: f32, // Distance from center
    pub target: Point3<f32>,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            theta: 0.8,
            phi: 1.2,
            radius: 80.0,
            target: Point3::origin(),
        }
    }

    /// Return the camera directions: (forward, right, up, eye_position)
    pub fn get_vectors(&self) -> (NVec3<f32>, NVec3<f32>, NVec3<f32>, Point3<f32>) {
        let x = self.radius * self.phi.sin() * self.theta.cos();
        let y = self.radius * self.phi.cos();
        let z = self.radius * self.phi.sin() * self.theta.sin();
        let eye = Point3::new(x, y, z);

        let forward = (self.target.coords - eye.coords).normalize();
        let world_up = NVec3::new(0.0f32, 1.0f32, 0.0f32);
        
        let right = forward.cross(&world_up).normalize();
        let up = right.cross(&forward).normalize();

        (forward, right, up, eye)
    }
}

pub struct RenderResources {
    pub pipeline: wgpu::RenderPipeline,
    pub uniforms_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl RenderResources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Lensing Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("lensing_shader.wgsl"))),
        });

        // Initialize Uniforms
        let uniforms = LensingUniforms {
            camera_pos: [0.0; 3],
            bh_mass: 10.0,
            camera_forward: [0.0; 3],
            rs: 20.0,
            camera_right: [0.0; 3],
            time: 0.0,
            camera_up: [0.0; 3],
            screen_width: 800.0,
            screen_height: 600.0,
            _pad: [0.0; 3],
        };

        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lensing Uniform Buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lensing Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lensing Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buffer.as_entire_binding(),
                }
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Lensing Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Lensing Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            uniforms_buffer,
            bind_group,
        }
    }

    pub fn update_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        camera: &Camera,
        sim: &SimulationState,
        width: f32,
        height: f32,
    ) {
        let (forward, right, up, eye) = camera.get_vectors();

        let uniforms = LensingUniforms {
            camera_pos: [eye.x, eye.y, eye.z],
            bh_mass: sim.black_hole.mass as f32,
            camera_forward: [forward.x, forward.y, forward.z],
            rs: sim.black_hole.schwarzschild_radius as f32,
            camera_right: [right.x, right.y, right.z],
            time: sim.time as f32,
            camera_up: [up.x, up.y, up.z],
            screen_width: width,
            screen_height: height,
            _pad: [0.0; 3],
        };

        queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));
    }
}
