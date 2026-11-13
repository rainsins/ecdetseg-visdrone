// Active training protocol: P150u. Historical PM2 configuration is in archive/pre_p150u/.
const path = require('path');
module.exports = {
  apps: [
    {
      name: 'p150u-all',
      script: 'scripts/run_all.sh',
      interpreter: 'bash',
      cwd: __dirname,
      autorestart: false,
      kill_timeout: 60000,
      stop_signal: 'SIGTERM',
      time: true,
      merge_logs: true,
      env: {
        VISDRONE_ROOT: process.env.VISDRONE_ROOT || '',
        WEIGHTS_DIR: process.env.WEIGHTS_DIR || '',
        EXP_CONFIG: process.env.EXP_CONFIG || 'configs/ecdet/ecdet_s_p150u.yml',
        EXP_OUT_ROOT: process.env.EXP_OUT_ROOT || path.join(__dirname, 'outputs', 'p150u'),
        EXP_RES: process.env.EXP_RES || '640',
        SEEDS: process.env.SEEDS || '42 43 44',
        INITS: process.env.INITS || 'e1 e2 e3',
        CONTINUE_ON_FAIL: process.env.CONTINUE_ON_FAIL || '0',
      },
    },
  ],
};
