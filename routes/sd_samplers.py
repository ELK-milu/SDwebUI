from collections import namedtuple


SamplerData = namedtuple('SamplerData', ['name', 'constructor', 'aliases', 'options'])

samplers_k_diffusion = [
    ('Euler a', 'sample_euler_ancestral', ['k_euler_a'], {}),
    ('Euler', 'sample_euler', ['k_euler'], {}),
    ('LMS', 'sample_lms', ['k_lms'], {}),
    ('Heun', 'sample_heun', ['k_heun'], {}),
    ('DPM2', 'sample_dpm_2', ['k_dpm_2'], {}),
    ('DPM2 a', 'sample_dpm_2_ancestral', ['k_dpm_2_a'], {}),
    ('DPM fast', 'sample_dpm_fast', ['k_dpm_fast'], {}),
    ('DPM adaptive', 'sample_dpm_adaptive', ['k_dpm_ad'], {}),
    ('LMS Karras', 'sample_lms', ['k_lms_ka'], {'scheduler': 'karras'}),
    ('DPM2 Karras', 'sample_dpm_2', ['k_dpm_2_ka'], {'scheduler': 'karras'}),
    ('DPM2 a Karras', 'sample_dpm_2_ancestral', ['k_dpm_2_a_ka'], {'scheduler': 'karras'}),
]


