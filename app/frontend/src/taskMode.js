const TEXT_TASK_MODES = new Set(['KIS', 'VQA', 'TRAKE']);

export function resolveRequestTask(taskMode, hasImageReference = false) {
  if (hasImageReference) return 'VKIS';
  return TEXT_TASK_MODES.has(taskMode) ? taskMode : 'KIS';
}
