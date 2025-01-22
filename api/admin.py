from django.contrib import admin
from django.utils.html import format_html
from .models import GeneratedAudio, SpeechRecognitionResult, GeneretedText

class GeneratedAudioAdmin(admin.ModelAdmin):
    list_display = ['text', 'phonemes', 'audio_player']

    def audio_player(self, obj):
        if obj.audio_file:
            return format_html('<audio controls><source src="{}" type="audio/wav"></audio>', obj.audio_file.url)
        return "No audio"

    audio_player.short_description = 'Audio'

class GeneratedAudioAdmin1(admin.ModelAdmin):
    list_display = ['recognized_text', 'audio_player']

    def audio_player(self, obj):
        if obj.audio_file:
            return format_html('<audio controls><source src="{}" type="audio/wav"></audio>', obj.audio_file.url)
        return "No audio"

    audio_player.short_description = 'Audio'

admin.site.register(GeneratedAudio, GeneratedAudioAdmin)

admin.site.register(SpeechRecognitionResult,GeneratedAudioAdmin1)

class GeneretedTextAdmin(admin.ModelAdmin):
    list_display = ['image_preview', 'text_snippet', 'created_at']

    def image_preview(self, obj):
        if obj.img:
            return format_html('<img src="{}" style="width: 60px; height: auto;" />', obj.img.url)
        return "No Image"

    image_preview.short_description = 'Image Preview'

    def text_snippet(self, obj):
        return obj.text[:50]  # Show the first 50 characters of the text

    text_snippet.short_description = 'Text Snippet'

# Register the model with the custom admin class
admin.site.register(GeneretedText, GeneretedTextAdmin)