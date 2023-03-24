document.addEventListener('DOMContentLoaded', function () {
  const resultDiv = document.getElementById('result');
  const pokemonNameSpan = document.getElementById('pokemon-name');

  const urlParams = new URLSearchParams(window.location.search);
  const pokemonName = urlParams.get('pokemon_name');

  if (pokemonName) {
    resultDiv.style.display = 'block';
    pokemonNameSpan.textContent = pokemonName;
  }
});
