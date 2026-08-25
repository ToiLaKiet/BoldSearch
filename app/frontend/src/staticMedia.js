export function apiOrigin(apiUrl = 'http://localhost:8000') {
  return String(apiUrl).replace(/\/api\/?$/, '').replace(/\/$/, '');
}


export function staticMediaUrl(path, baseUrl = '') {
  const value = String(path || '');
  if (!value || /^[a-z][a-z\d+.-]*:/i.test(value)) return value;
  const normalizedBaseUrl = String(baseUrl).replace(/\/$/, '');
  if (!normalizedBaseUrl) return value;
  return `${normalizedBaseUrl}${value.startsWith('/') ? value : `/${value}`}`;
}
