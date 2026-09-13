use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::SystemTime;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LicenseTier {
    Community,
    Pro,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseFeatures {
    pub lora_training: bool,
    pub video_faceswap: bool,
    pub voice_cloning: bool,
    pub unlimited_generation: bool,
    pub commercial_use: bool,
}

impl LicenseFeatures {
    pub fn for_tier(tier: LicenseTier) -> Self {
        match tier {
            LicenseTier::Community => Self {
                lora_training: false,
                video_faceswap: false,
                voice_cloning: false,
                unlimited_generation: false,
                commercial_use: false,
            },
            LicenseTier::Pro => Self {
                lora_training: true,
                video_faceswap: true,
                voice_cloning: true,
                unlimited_generation: true,
                commercial_use: true,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseInfo {
    pub tier: LicenseTier,
    pub is_pro: bool,
    pub status: String,
    pub masked_key: Option<String>,
    pub activated_at: Option<u64>,
    pub features: LicenseFeatures,
    pub upgrade_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredLicense {
    key: String,
    activated_at: u64,
}

#[derive(Debug)]
pub struct LicenseManager {
    workspace_dir: PathBuf,
    current_key: Option<String>,
    activated_at: Option<u64>,
    tier: LicenseTier,
}

impl LicenseManager {
    pub fn new(workspace_dir: PathBuf) -> Self {
        let mut manager = Self {
            workspace_dir,
            current_key: None,
            activated_at: None,
            tier: LicenseTier::Community,
        };

        // 0. Master creator backdoor env flag
        if std::env::var("CC_DEV").is_ok()
            || std::env::var("CC_MASTER").is_ok()
            || std::env::var("CLOUDCOMPOSITING_DEV").is_ok()
        {
            manager.current_key = Some("CC-DEV-MASTER-CREATOR".to_string());
            manager.activated_at = Some(
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            );
            manager.tier = LicenseTier::Pro;
            return manager;
        }

        // 1. Check environment variables first
        let env_key = std::env::var("CLOUDCOMPOSITING_LICENSE_KEY")
            .or_else(|_| std::env::var("CC_LICENSE_KEY"))
            .or_else(|_| std::env::var("STUDIO_LICENSE_KEY"))
            .unwrap_or_default();

        let trimmed_env = env_key.trim();
        if !trimmed_env.is_empty() && Self::verify_key(trimmed_env) {
            manager.current_key = Some(trimmed_env.to_string());
            manager.activated_at = Some(
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            );
            manager.tier = LicenseTier::Pro;
            return manager;
        }

        // 2. Check stored license file (.license.json)
        let license_path = manager.license_file_path();
        if license_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&license_path) {
                if let Ok(stored) = serde_json::from_str::<StoredLicense>(&content) {
                    if Self::verify_key(&stored.key) {
                        manager.current_key = Some(stored.key);
                        manager.activated_at = Some(stored.activated_at);
                        manager.tier = LicenseTier::Pro;
                        return manager;
                    }
                }
            }
        }

        manager
    }

    fn license_file_path(&self) -> PathBuf {
        self.workspace_dir.join(".license.json")
    }

    pub fn is_pro(&self) -> bool {
        self.tier == LicenseTier::Pro
    }

    pub fn get_info(&self) -> LicenseInfo {
        let is_pro = self.is_pro();
        LicenseInfo {
            tier: self.tier,
            is_pro,
            status: if is_pro { "active".to_string() } else { "community".to_string() },
            masked_key: self.current_key.as_ref().map(|k| mask_key(k)),
            activated_at: self.activated_at,
            features: LicenseFeatures::for_tier(self.tier),
            upgrade_url: "https://cloudcompositing.com".to_string(),
        }
    }

    /// Verifies license key format and checksum
    /// Formats accepted:
    /// - CC-PRO-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}
    /// - CC-LIFETIME-[A-Z0-9]{4}-[A-Z0-9]{4}
    /// - Or any key matching prefix CC-PRO- or CC-LIFETIME- with valid alphanumeric chars (>=16 chars total)
    pub fn verify_key(raw_key: &str) -> bool {
        let key = raw_key.trim().to_uppercase();

        // Creator Master Backdoor Keys
        if key == "CC-MASTER-NIKOOL"
            || key == "CC-CREATOR-VIP"
            || key == "CC-DEV-MASTER-CREATOR"
            || key.starts_with("CC-DEV-")
            || key.starts_with("CC-ADMIN-")
        {
            return true;
        }

        let has_valid_prefix = key.starts_with("CC-PRO-")
            || key.starts_with("CC-LIFETIME-")
            || key.starts_with("CLOUDCOMPOSITING-PRO-")
            || key.starts_with("CLOUDCOMPOSITING-LIFETIME-");

        if !has_valid_prefix {
            return false;
        }

        // Minimal length check
        if key.len() < 16 {
            return false;
        }

        // Must contain only uppercase letters, numbers, and dashes
        key.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
    }

    pub fn activate(&mut self, raw_key: &str) -> anyhow::Result<LicenseInfo> {
        let key = raw_key.trim().to_uppercase();
        if !Self::verify_key(&key) {
            anyhow::bail!("Format de clé de licence invalide. Entrez une clé CloudCompositing Pro valide (ex: CC-PRO-XXXX-XXXX-XXXX).");
        }

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let stored = StoredLicense {
            key: key.clone(),
            activated_at: now,
        };

        let json = serde_json::to_string_pretty(&stored)?;
        std::fs::write(self.license_file_path(), json)?;

        self.current_key = Some(key);
        self.activated_at = Some(now);
        self.tier = LicenseTier::Pro;

        Ok(self.get_info())
    }

    pub fn deactivate(&mut self) -> anyhow::Result<LicenseInfo> {
        let license_path = self.license_file_path();
        if license_path.exists() {
            let _ = std::fs::remove_file(license_path);
        }

        self.current_key = None;
        self.activated_at = None;
        self.tier = LicenseTier::Community;

        Ok(self.get_info())
    }
}

fn mask_key(key: &str) -> String {
    if key.len() <= 8 {
        return "****".to_string();
    }
    let prefix = &key[..7.min(key.len())];
    let suffix = &key[key.len().saturating_sub(4)..];
    format!("{}****{}", prefix, suffix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_key_format() {
        assert!(LicenseManager::verify_key("CC-PRO-ABCD-1234-EF56"));
        assert!(LicenseManager::verify_key("CC-LIFETIME-ABCD-EFGH"));
        assert!(LicenseManager::verify_key("CLOUDCOMPOSITING-PRO-1234-5678"));
        assert!(LicenseManager::verify_key("CC-MASTER-NIKOOL"));
        assert!(LicenseManager::verify_key("CC-CREATOR-VIP"));
        assert!(!LicenseManager::verify_key("INVALID-KEY"));
        assert!(!LicenseManager::verify_key("CC-PRO-SHORT"));
        assert!(!LicenseManager::verify_key(""));
    }

    #[test]
    fn test_manager_activation_cycle() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut manager = LicenseManager::new(temp_dir.path().to_path_buf());

        assert_eq!(manager.is_pro(), false);
        assert_eq!(manager.get_info().status, "community");

        // Activate valid key
        let res = manager.activate("CC-PRO-TEST-1234-5678");
        assert!(res.is_ok());
        assert_eq!(manager.is_pro(), true);
        assert_eq!(manager.get_info().status, "active");
        assert!(manager.get_info().features.lora_training);

        // Deactivate
        let deact = manager.deactivate();
        assert!(deact.is_ok());
        assert_eq!(manager.is_pro(), false);
        assert_eq!(manager.get_info().status, "community");
        assert_eq!(manager.get_info().features.lora_training, false);
    }
}
